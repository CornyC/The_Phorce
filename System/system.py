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
from openmm import unit
from math import sqrt


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
    MDA_reader_object : Coord_Toolz.mdanalysis.MDA_reader object
        Contains atom and residue information
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
        Unit: kJ/mol/nm
    qm_energies: numpy array
        potential energies of each sampled structure evaluated by the quantum chemical method
        Unit: kJ/mol
    mm_forces: numpy array
        forces on each atom of the ini_coords evaluated by the classical MD method for each sampled structure. 
        Shape: (n_conformations, n_atoms, 3)
        Unit: kJ/mol/nm
    mm_energies: numpy array
        potential energies of each sampled structure evaluated by the classical MD method. Shape: (n_conformations)
        Unit: kJ/mol
    eqm_bsse: numpy array
        Basis set superposition error
    ...net... : numpy array
        subtracted net property (e.g. between 2 molecules or molecule and solvent)
    weights : float
        weights to weigh conformations
    dupes : dict of dict of list of ints
        e.g. {'all': {'OG311': [3, 4, 14, 15], 'OG2P1': [5], 'OG303': [1, 12]},
              'mol2': {'OG311': [3, 4], 'OG2P1': [5], 'OG303': [1]},
              'mol1': {'OG311': [3, 4], 'OG2P1': [5], 'OG303': [1]}} 
        containing duplicate atom types and their atom indices
    nbfix_dupes : dict of list of lists
        e.g. {'all': [[0,1]], 'nosol': [[0,1]]}
        contains indices from openmm_systems[sys_type].nbfix where the same nbfix/ same atom pair is listed.
    interaction_dupes : nested dict
        contains atom index combinations consisting of duplicate atom types only and their positions (indices) in ff_optimizable[sys_type][force_group]
    acoef/bcoef : dict
        has same keys as openmm_systems; holds acoeffs and bcoeffs which are needed for OpenMM's NBFIX handling as CustomNonbondedForce.
    hybrid : bool
        default = False, flag set True if one subsystem has a NBFIX and one doesn't.
    hybrid_sys_types : dict
        specifies which subsystems need which type of nonbonded parameter processing. e.g. {'all': True,
                                                                                             'mol1': False,
                                                                                             'mol2': True} 
                                                                    meaning True has NBFIX, False doesn't.
    hybrid_check_performed : bool
        default = False, signals whether system has been checked for hybrid nonbonded parameter processing
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
        self.MDA_reader_object = None
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
        self.hybrid = False
        self.hybrid_sys_types = None
        self.hybrid_check_performed = False

        self.slice_list = None
        self.dupes = None
        self.interaction_dupes = None
        self.reduced_indexed_ff_optimizable = None
        self.reduced_ff_optimizable_values = None
        self.vectorized_reduced_ff_optimizable_values = None
        self.vectorized_parameters = None
        self.scaling_factors = None
        self.scaled_parameters = None

    def set_ini_coords(self, MDA_reader_object):
        """
        reads in the initial atoms & coordinates from file(s) using the MDA_reader, stores them in Molecular_system,
        and acquires properties derived from them based on the settings in Molecular_system.  

        Parameters
        ----------
        MDA_reader_object: Coord_Toolz.mdanalysis.MDA_reader object
            Contains MDA Universe and atomgroups

        sets:
            self.MDA_reader_object
            self.n_conformations
            self.n_atoms
            self.ini_coords
        """

        self.MDA_reader_object = MDA_reader_object
        coords = ct.get_coords(self.MDA_reader_object.universes['all'].atoms)
        self.n_conformations = len(coords) 
        self.n_atoms['all'] = len(self.MDA_reader_object.universes['all'].atoms)

        self.ini_coords['all'] = coords

        assert self.paths is not None, 'Molecular_system.paths not set'

        assert self.paths.mm_traj is not None, 'Conformations trajectory not found.'

        assert self.paths.mm_crd is not None, 'Coordinate file not found.'

        assert self.paths.mm_top is not None, 'Topology fiel not found.'

        if self.MDA_reader_object.universes['nosol'] is not None:

            nosol_coords = ct.get_coords(self.MDA_reader_object.universes['nosol'])
            self.ini_coords['nosol'] = nosol_coords
            self.n_atoms['nosol'] = len(self.MDA_reader_object.universes['nosol'])

            assert self.paths.mm_nosol_crd is not None, 'Coordinate file for molecule '\
                                                        'w/o solvent not found.'
            assert self.paths.mm_nosol_top is not None, 'Topology file for molecule '\
                                                        'w/o solvent not found.'

        elif self.MDA_reader_object.universes['mol1'] is not None:

            mol1_coords = ct.get_coords(self.MDA_reader_object.universes['mol1'])
            self.ini_coords['mol1'] = mol1_coords
            self.n_atoms['mol1'] = len(self.MDA_reader_object.universes['mol1'])
            mol2_coords = ct.get_coords(self.MDA_reader_object.universes['mol2'])          
            self.ini_coords['mol2'] = mol2_coords
            self.n_atoms['mol2'] = len(self.MDA_reader_object.universes['mol2'])

            assert self.paths.mm_mol1_crd is not None, 'Coordinate file for molecule 1 '\
                                                        'w/o molecule 2 not found.'
            assert self.paths.mm_mol1_top is not None, 'Topology file for molecule '\
                                                        'w/o molecule 2 not found.'
            assert self.paths.mm_mol2_crd is not None, 'Cordinate file for molecule 2 '\
                                                        'w/o molecule 1 not found'
            assert self.paths.mm_mol2_top is not None, 'Topology file for molecule 2 '\
                                                        'w/o molecule 1 not found.'
            

    def read_qm_charges_energies_forces_optcoords(self, sys_type: str, path: str, cp2k_outfilename: str, filename=None, charge_type=None):

        """
        Useful if cp2k calculations have been run externally (e.g. on a cluster).
        reads cp2k .out files and extracts charges, energies, and forces from it

        Parameters
        ----------
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 
        path : str
            path to externally cp2k optimized structures
        cp2k_outfilename : str
            equal to <cp2k>.out
        filename : str
            default = None. Needed if sys_type = 'all'
            equal to &GLOBAL>PROJECT str in cp2k.inp file. Required to read optimized coords.
        charge_type : str
            default = None. If charges should be read, supply one of the following: 
            'Mulliken', 'Hirshfeld', 'RESP'

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
        if sys_type == 'all':
            self.opt_coords[sys_type] = np.zeros(self.ini_coords[sys_type].shape)

        self.qm_forces[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.qm_energies[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))

        if charge_type != None:
            self.qm_charges[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1]))

        #### fill the arrays ####

        for n in range(self.n_conformations): 
            
            assert Path(path+'frame'+str(n)+'/'+cp2k_outfilename+'.out').is_file() is True, 'file {} does not exist'.format(path+'frame'+str(n)+'/'+cp2k_outfilename+'.out')

            print('reading qm frame '+str(n)+' info')
            
            read = read_external_file(path+'frame'+str(n)+'/', cp2k_outfilename+'.out')
            
            energy, forces = read_qm_energy_forces(read, n, cp2k_outfilename, path, self.ini_coords[sys_type])
            
            self.qm_forces[sys_type][n, :, :] = forces * 2.62549961709828E+03 / 0.0529177249 #Hartree/Bohr to kJ/mol/nm
            self.qm_energies[sys_type][n] = energy * 2.62549961709828E+03 #Hartree to kJ/mol

            if charge_type != None:

                charges = read_qm_charges(read, charge_type, path, n, cp2k_outfilename, self.ini_coords[sys_type])
                self.qm_charges[sys_type][n, :] = charges

            if sys_type == 'all':

                u = mda.Universe(path + 'frame' + str(n) + '/' + filename + '-pos-1.xyz')
                coords = ct.get_coords(u.atoms)[-1]
                self.opt_coords[sys_type][n, :, :] = coords


    def read_ase_energy_forces(self, sys_type: str, path: str, atomgroup_name: str):

        """
        Useful if ASE-cp2k calculations have been run externally (e.g. on a cluster).
        reads ASE-generated cp2k.out files and extracts energies and forces from it

        Parameters
        ----------
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 
        path : str
            path to externally optimized structures
        atomgroup_name : str
            e.g. 'mol1' or 'nosol'
        self.ini_coords : unoptimized coordinates of conformations
        self.n_conformations : number of conformations
        
        sets:
            self.qm_forces : quantum forces
            self.qm_energies : quantum energies
        """

        if path[-1] != '/':
            path += '/'        
    
        # init arrays
        self.qm_forces[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.qm_energies[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))

        for n in range(self.n_conformations): 

            assert Path(path + 'frame' + str(n) + '/forces_energy_'+atomgroup_name+'_frame'+str(n)+'.txt').is_file() is True, 'file {} does not exist'.format(path + 'frame' + str(n) + '/forces_energy_'+atomgroup_name+'_frame'+str(n)+'.txt')

            forces = np.genfromtxt(path + 'frame' + str(n) + '/forces_energy_'+atomgroup_name+'_frame'+str(n)+'.txt', skip_header=1)

            f = open(path + 'frame' + str(n) + '/forces_energy_'+atomgroup_name+'_frame'+str(n)+'.txt', 'r')
            read = []
            for i, line in enumerate(f.readlines()):
                line = line.strip('# E:\n')
                if i == 0:
                    read.append(line)
            f.close()
            energy = float(read[0])

            self.qm_forces[sys_type][n, :, :] = forces 
            self.qm_energies[sys_type][n] = energy 


    def collect_ase_settings(self, n_threads, cell_dimensions, pbc, input_control, run_type):

        """
        Stores ASE settings in system.

        Parameters
        ----------
        n_threads : int
            number of OpenMP threads
        cell_dimensions : np.array
            defines the unit cell vectors: ([x, y, z])
        pbc : np.array
            defines which direction is supposed to be periodic, e.g. ([True, True, True]) for periodicity in all directions
        input_control : str
            sets the input commands to the ASE calculator (e.g. like a cp2k.inp file)
        run_type : str
            'single_point' or 'optimization'
        """

        self.n_threads = n_threads
        self.ase_cell_dims = cell_dimensions
        self.ase_pbc = pbc
        self.ase_input_control = input_control
        self.ase_run_type = run_type


    def _run_ase(self, coords, atomgroup, atomgroup_name, sys_type):
        """
        Loops over conformations and runs ASE to obtain QM data.

        Parameters
        ----------
        coords : np.array
            molecule coordinates as numpy array
        atomgroup : MDAnalysis AtomGroup object
            e.g. mol1
        atomgroup_name : str
            e.g. 'mol1', needed for output files
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 

        internal parameters : self.n_threads, self.ase_cell_dims, self.ase_pbc, self.ase_input_control, self.ase_run_type

        sets:
            self.qm_forces[sys_type] : quantum forces of atomgroup after geometry opt
            self.qm_energies[sys_type] : quantum energies of atomgroup after geom opt
            (self.opt_coords[sys_type] : coordinates after DFT geometry optimization)
        """
        for setting in [self.ase_cell_dims, self.ase_pbc, self.ase_input_control, self.n_threads, self.ase_run_type]:
            try: 
                assert setting is not None, 'ASE settings are not defined, call collect_ase_settings(n_threads, cell_dimensions, pbc, input_control, run_type) first'
            except NameError:
                pass

        for frame_nr, frame in enumerate(coords):

            os.system('mkdir frame'+str(frame_nr))
            os.chdir('frame'+str(frame_nr))

            ase_sys = ASE_system(atomgroup.elements, frame)
            ase_sys.cell = self.ase_cell_dims
            ase_sys.pbc = self.ase_pbc
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
                        inp=self.ase_input_control,
                        print_level='MEDIUM',
                        command = 'env OMP_NUM_THREADS='+str(self.n_threads)+' cp2k_shell.ssmp')

            ase_sys.atoms.calc = calc
            if sys_type is not 'all':
                self.ase_run_type = 'single_point'
            ase_sys.run_calculation(run_type = self.ase_run_type)
            np.savetxt('forces_energy_' + atomgroup_name + '_frame' + str(frame_nr) + '.txt', ase_sys.forces,
                       header='E: ' + str(ase_sys.energy))
            outstr = 'cp cp2k.out ' + atomgroup_name + '_frame' + str(frame_nr) + '.out'
            os.system(outstr)

            # grab forces
            self.qm_forces[sys_type][frame_nr, :, :] = ase_sys.forces #kJ/mol/nm

            # grab energies
            self.qm_energies[sys_type][frame_nr] = ase_sys.energy #kJ/mol

            if self.ase_run_type == 'optimization':
                # grab optcoords
                self.opt_coords[sys_type][frame_nr, :, :] = ase_sys.opt_coords

            os.system('rm cp2k.out')
            os.system('pkill cp2k_shell.ssmp')
            os.system('touch cp2k.out')
            os.chdir('..') 


    def generate_qm_data_w_ase(self, sys_type: str, paths: str, atomgroup, atomgroup_name: str):
        """
        Uses ASE to generate QM data- Only energies, forces, and optimized coordinates are supported. If you want charges, use generate_qm_data_w_cp2k()

        Parameters
        ----------
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 
        paths : paths.Paths object
            contains working dir n stuff
        paths : paths.Paths object
            contains working dir n stuff
        atomgroup : MDAnalysis AtomGroup object
            e.g. mol1
        atomgroup_name : str
            e.g. 'mol1', needed for output files
        """

        #### init arrays ####
        self.qm_forces[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.qm_energies[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))
        if self.ase_run_type == 'optimization':
            self.opt_coords[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))

        if Path(paths.working_dir + paths.project_name).is_dir() is True:

            os.chdir(paths.working_dir + paths.project_name)

        else:

            os.chdir(paths.working_dir)
            os.system('mkdir '+paths.project_name)
            os.chdir(paths.project_name)

        coords = ct.get_coords(atomgroup)

        self._run_ase(coords, atomgroup, atomgroup_name, sys_type)


    def collect_cp2k_settings(self, cp2k_input: str, n_threads: int, cp2k_binary_name: str, cp2k_run_type: str):
        """
        Stores cp2k settings in system.

        Parameters
        ----------
        cp2k_input : str
            all the commands and strings that go into a cp2k .inp file (see cp2k doc). It is recommended to load the
            input into a separate variable beforehand
        n_threads : int
            number of OpenMP threads to use for the parallel cp2k calculation
        cp2k_binary_name : str
            name by which the cp2k binary is called on your machine
        cp2k_run_type : str
            'single_point' or 'optimization'
        """
        self.cp2k_input = cp2k_input
        self.n_threads = n_threads
        self.cp2k_binary_name = cp2k_binary_name
        self.cp2k_run_type = cp2k_run_type


    def generate_qm_data_w_cp2k(self, sys_type: str, paths: str, MDA_reader, charge_type=None):
        """
        Calculates QM energies, forces (and charges) using native cp2k. 
        Requires topology file and starting coords as well as a valid cp2k input file (please check cp2k doc).

        Parameters
        ----------
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 
        paths : paths.Paths object
            contains paths and filenames etc
        MDA_reader : Coord_Toolz.mdanalysis.MDA_reader object
            contains MDA universes w/ atom info, coords, ...
        charge_type : str
            Default=None, no charges will be computed.
            Charge options are 'Mulliken', 'Hirshfeld', 'RESP'
        """

        for setting in [self.cp2k_input, self.n_threads, self.cp2k_binary_name, self.cp2k_run_type]:
            try:
                assert setting is not None, 'cp2k settings not defined, run collect_cp2k_settings(cp2k_input, n_threads, cp2k_binary_name, cp2k_run_type, charge_type) first.'
            except NameError:
                pass

        from ..Direct_cp2k_calculation.direct_cp2k import Direct_Calculator as cc

        #### init arrays ####
        self.qm_forces[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.qm_energies[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))
        if self.cp2k_run_type == 'optimization':
            self.opt_coords[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))

        #### fill the arrays #### 
        for n in range(self.n_conformations):

            #### construct charge calculator object ####
            cp2k_calc = cc(paths, self.cp2k_run_type)

            cp2k_calc.generate_cp2k_input_file(self.cp2k_input)

            os.system('mkdir frame'+str(n))
            os.chdir('frame'+str(n))
            os.system('cp '+paths.mm_top+' .') # copy the topol file
            os.system('cp '+cp2k_calc.project_path+cp2k_calc.project_name+cp2k_calc.run_type+'.inp .')
            MDA_reader.universes[sys_type].atoms.write('frame'+str(n)+'.pdb', frames=MDA_reader.universes[sys_type].trajectory[n:n+1])

            path = os.getcwd()

            # run the calculation
            cp2k_calc.run_cp2k(self.n_threads, self.cp2k_binary_name, path)

            # grab results
            read = read_external_file(cp2k_calc.project_path+'frame'+str(n)+'/', cp2k_calc.project_name+run_type+'.out')

            if charge_type is not None:
                cp2k_calc.charges = read_qm_charges(read, charge_type, cp2k_calc.project_path, n, cp2k_calc.project_name+run_type, self.ini_coords)
                self.qm_charges[sys_type][n, :] = cp2k_calc.charges

            cp2k_calc.extract_energy(read)
            self.qm_energies[sys_type][n] = cp2k_calc.energy * 2.62549961709828E+03 #Hartree to kJ/mol

            cp2k_calc.extract_forces(self.ini_coords[sys_type].shape[1], read)
            self.qm_forces[sys_type][n, :, :] = cp2k_calc.forces * 2.62549961709828E+03 / 0.0529177249 #Hartree/Bohr to kJ/mol/nm

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
                    mask1 = np.invert(np.isin(self.MDA_reader_object.universes['mol1'], self.MDA_reader_object.universes['mol2']))
                    mask2 = np.invert(np.isin(self.MDA_reader_object.universes['mol2'], self.MDA_reader_object.universes['mol1']))

                    mol1_slice = self.MDA_reader_object.universes['mol1']._ix[mask1]
                    mol2_slice = self.MDA_reader_object.universes['mol2']._ix[mask2]
                    mol12_slice = np.concatenate((mol1_slice, mol2_slice))
                    mol1_atom_numbers = np.arange(0,len(self.MDA_reader_object.universes['mol1']),1,dtype='int')[mask1]
                    mol2_atom_numbers = np.arange(0,len(self.MDA_reader_object.universes['mol2']),1,dtype='int')[mask2]

                    self.qm_net_forces = self.qm_forces['all'][:,mol12_slice,:] - np.concatenate((self.qm_forces['mol1'][:,mol1_atom_numbers,:], self.qm_forces['mol2'][:,mol2_atom_numbers,:]), axis = 1)
                    
        else:
            mask = np.isin(self.MDA_reader_object.universes['all'].atoms, self.MDA_reader_object.universes['nosol'])
            mol_slice = self.MDA_reader_object.universes['all'].atoms._ix[mask]

            self.qm_net_forces = self.qm_forces['all'][:,mol_slice,:] - self.qm_forces['nosol']


    def generate_mm_energies_forces(self, sys_type, verbose=False): 
        """
        collects the OpenMM system's classical properties (charges,energies&forces)

        Parameters
        ----------
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 
        verbose : bool
            default=False, prints info if set True

        other (internal) params:
            
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
            self.mm_forces[sys_type][frame_nr, :, :] = forces #kJ/mol/nm
        
            # grab energies
            self.mm_energies[sys_type][frame_nr] = epot #kJ/mol

        if verbose == True:

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
                    mask1 = np.invert(np.isin(self.MDA_reader_object.universes['mol1'], self.MDA_reader_object.universes['mol2']))
                    mask2 = np.invert(np.isin(self.MDA_reader_object.universes['mol2'], self.MDA_reader_object.universes['mol1']))

                    mol1_slice = self.MDA_reader_object.universes['mol1']._ix[mask1]
                    mol2_slice = self.MDA_reader_object.universes['mol2']._ix[mask2]
                    mol12_slice = np.concatenate((mol1_slice, mol2_slice))
                    mol1_atom_numbers = np.arange(0,len(self.MDA_reader_object.universes['mol1']),1,dtype='int')[mask1]
                    mol2_atom_numbers = np.arange(0,len(self.MDA_reader_object.universes['mol2']),1,dtype='int')[mask2]

                    self.mm_net_forces = self.mm_forces['all'][:,mol12_slice,:] - np.concatenate((self.mm_forces['mol1'][:,mol1_atom_numbers,:], self.mm_forces['mol2'][:,mol2_atom_numbers,:]), axis = 1)
                    
        else:
            mask = np.isin(self.MDA_reader_object.universes['all'].atoms, self.MDA_reader_object.universes['nosol'])
            mol_slice = self.MDA_reader_object.universes['all'].atoms._ix[mask]

            self.mm_net_forces = self.mm_forces['all'][:,mol_slice,:] - self.mm_forces['nosol']



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


    def _check_for_hybrid_nb_processing(self):
        """
        It can happen that based on the user's atom selection, the 'all' system has an NBFIX and triggers OpenMM's CustomNonbondedForce 
        processing while in the 'nosol' or 'mol1/mol2' subsystems no NBFIX is present and nonbonded parameters go through the regular processing.
        This function sets a flag (self.hybrid) that triggers hybrid parameter processing during the pushback of the parameters from the optimizer back into the 
        OpenMM Context for computing net properties. 

        (internal) parameters:
            self.openmm_systems
        sets:
            self.hybrid
            self.hybrid_sys_types
        """
        print('performing hybrid check...')

        hybrid_sys_types = {}

        nb_processing_type = []

        hybrid = False

        for sys_type in self.openmm_systems:

            if self.openmm_systems[sys_type] != None:

                for nb_type in self.openmm_systems[sys_type].ff_optimizable:

                    nb_processing_type.append(nb_type)

        if 'CustomNonbondedForce' in nb_processing_type: #check if CustomNonbondedForce is present at all

            custom_nb_counter = 0
            sys_type_counter = 0

            for sys_type in self.openmm_systems:

                if self.openmm_systems[sys_type] != None:

                    sys_type_counter += 1

                    if 'CustomNonbondedForce' in self.openmm_systems[sys_type].ff_optimizable:

                        custom_nb_counter += 1

            if custom_nb_counter != sys_type_counter: #check if all subsystems have NBFIX

                hybrid = True 

                print('hybrid sys found')

        if hybrid == True: #find which ones have NBFIX and which don't

            for sys_type in self.openmm_systems:

                if self.openmm_systems[sys_type] != None:

                    if 'CustomNonbondedForce' in self.openmm_systems[sys_type].ff_optimizable:

                        hybrid_sys_types[sys_type] = True

                    else: 
                        hybrid_sys_types[sys_type] = False
                        
            self.hybrid = True
            self.hybrid_sys_types = hybrid_sys_types

            print('hybrid set to True')

        self.hybrid_check_performed = True


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

    
    def calculate_acoef_bcoef(self):
        """
        OpenMM NBFIX handling: CustomNonbondedForce('(a/r6)^2-b/r6; r6=r^6;'
                                                    'a=acoef(type1, type2);'
                                                    'b=bcoef(type1, type2)')
        First, rij and wdij are calculated for the atom types: rij = sigma_i + sigma_j and wdij = epsilon_i^(1/2) * epsilon_j
        Then, acoef = wdij^(1/2) * rij^6 and bcoef = 2 * wdij * rij^6
        WARNING: Static implementation for only one CustomNonbondedForce in the OpenMM system. If there are more, this will not work.
        
        internal parameters:
        self.openmm_systems[sys_type].custom_nb_params : np.array
            contains sigma, epsilon, atom type (in GMX/OMM units)
        self.openmm_systems[sys_type].nbfix : np.array
            contains NBFIX rij and wdij, atom type 1, atom type 2 (in GMX/OMM units)

        sets:
            self.openmm_systems[sys_type].ff_optimizable['CustomNonbondedForce']
        """
        for sys_type in self.openmm_systems.keys():

            if self.openmm_systems[sys_type] is not None:

                if self.hybrid is False:

                    num_lj_types = len(self.openmm_systems[sys_type].lj_type_list) 
                    acoef = [0 for i in range(num_lj_types*num_lj_types)]
                    bcoef = acoef[:]

                    for nbfix in self.openmm_systems[sys_type].nbfix:

                        for i in range(num_lj_types):

                            atomtype_i = self.openmm_systems[sys_type].custom_nb_params[i][-1]

                            for j in range(num_lj_types):

                                atomtype_j = self.openmm_systems[sys_type].custom_nb_params[j][-1]

                                if (nbfix[-2] == atomtype_i and nbfix[-1] == atomtype_j) == True:

                                    rij = nbfix[0]
                                    wdij = nbfix[1]

                                else:
                                    rij = self.openmm_systems[sys_type].custom_nb_params[i][0] + \
                                        self.openmm_systems[sys_type].custom_nb_params[j][0]
                                    
                                    wdij = sqrt(self.openmm_systems[sys_type].custom_nb_params[i][1] * \
                                            self.openmm_systems[sys_type].custom_nb_params[j][1])

                                acoef[i+num_lj_types*j] = sqrt(wdij) * rij**6
                                bcoef[i+num_lj_types*j] = 2 * wdij * rij**6

                    self.openmm_systems[sys_type].ff_optimizable['CustomNonbondedForce'][0]['acoef'] = acoef
                    self.openmm_systems[sys_type].ff_optimizable['CustomNonbondedForce'][0]['bcoef'] = bcoef

                if self.hybrid is True:

                    if self.hybrid_sys_types[sys_type] is True: # has NBFIX

                        num_lj_types = len(self.openmm_systems[sys_type].lj_type_list) 
                        acoef = [0 for i in range(num_lj_types*num_lj_types)]
                        bcoef = acoef[:]

                        for nbfix in self.openmm_systems[sys_type].nbfix:

                            for i in range(num_lj_types):

                                atomtype_i = self.openmm_systems[sys_type].custom_nb_params[i][-1]

                                for j in range(num_lj_types):

                                    atomtype_j = self.openmm_systems[sys_type].custom_nb_params[j][-1]

                                    if (nbfix[-2] == atomtype_i and nbfix[-1] == atomtype_j) == True:

                                        rij = nbfix[0]
                                        wdij = nbfix[1]

                                    else:
                                        rij = self.openmm_systems[sys_type].custom_nb_params[i][0] + \
                                            self.openmm_systems[sys_type].custom_nb_params[j][0]
                                        
                                        wdij = sqrt(self.openmm_systems[sys_type].custom_nb_params[i][1] * \
                                                self.openmm_systems[sys_type].custom_nb_params[j][1])

                                    acoef[i+num_lj_types*j] = sqrt(wdij) * rij**6
                                    bcoef[i+num_lj_types*j] = 2 * wdij * rij**6

                        self.openmm_systems[sys_type].ff_optimizable['CustomNonbondedForce'][0]['acoef'] = acoef
                        self.openmm_systems[sys_type].ff_optimizable['CustomNonbondedForce'][0]['bcoef'] = bcoef

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
    

    def reduce_ff_optimizable(self, slice_list): #TODO: needs to know if nb processing is hybrid! -> if true, compare cnb & nb & reduce
        """
        extracts force field parameters from ff_optimizable based on an atom selection broadcasted through slice_list. 
        Also eliminates 'duplicates' of the same atom type.
        WARNING: Static implementation for only one CustomNonbondedForce in the OpenMM system. If there are more, this will not work.

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
            self.openmm_systems[sys_type].custom_nb_params
            self.openmm_systems[sys_type].nbfix
        
        sets:
            self.reduced_indexed_ff_optimizable[sys_type][force_group] : dict of dict of np.arrays containing atom indices, atom types, and ff term values (nonbonded)
                or line indices, atom indices, and ff term values (bonded); not mutable
            self.reduced_ff_optimizable_values[sys_type][force_group] : dict of dict of np.arrays containing only the ff term values (nonbonded or bonded); mutable
                units: sigma in nm, epsilon in kJ/mol. NBFIX: rmin in nm, epsilon in kJ/mol
            self.dupes : dict of dict of atom types (str) and their inidces (-> marks duplicate atom types)
            self.nbfix_dupes : dict of list of indices indicating same atom pair/same NBFIX
        """

        reshape_column_indices = {'HarmonicBondForce': 4,
                    'HarmonicAngleForce': 5,
                    'PeriodicTorsionForce': 7,
                    'NonbondedForce': 3,
                    'NBException': 5}

        self.slice_list = slice_list
        
        # define and init the dictionaries by sys_type
        self.reduced_indexed_ff_optimizable = {k: [] for k in sorted(set(self.slice_list.keys())) if self.slice_list[k] != None} 
        self.reduced_ff_optimizable_values = {k: [] for k in sorted(set(self.slice_list.keys())) if self.slice_list[k] != None}
        self.dupes = {k: [] for k in sorted(set(self.slice_list.keys())) if self.slice_list[k] != None}
        self.nbfix_dupes = {k: [] for k in sorted(set(self.slice_list.keys())) if self.slice_list[k] != None}
        self.interaction_dupes = {k: {} for k in sorted(set(self.slice_list.keys())) if self.slice_list[k] != None}
        self.to_be_removed = {k: [] for k in sorted(set(self.slice_list.keys())) if self.slice_list[k] != None}

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

                        if 'CustomNonbondedForce' in list(self.openmm_systems[sys_type].ff_optimizable.keys()): #NBFIX
                            # remove useless sigma and epsilon columns
                
                            reduced_indexed_ff_opt = reduced_indexed_ff_opt[:,:-2]
                            reduced_ff_opt_values = reduced_ff_opt_values[:,:-2]

                            if sys_type == 'all':

                                if self.hybrid_check_performed == False:

                                    print('hybrid check not performed')

                                    self._check_for_hybrid_nb_processing()

                        # store in Molsys
                        self.reduced_indexed_ff_optimizable[sys_type][force_group].append(reduced_indexed_ff_opt)
                        self.reduced_ff_optimizable_values[sys_type][force_group].append(reduced_ff_opt_values) 

                    elif force_group in ['NBException', 'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']: 

                        sliceable_ff_optimizable = _array.view('<f8').reshape(len(_array), reshape_column_indices[force_group]) # turns 1d recarray into sliceable nd array 
                        reduced_indexed_ff_opt, reduced_ff_opt_values = self._eliminate_duplicate_atomtypes(sliceable_ff_optimizable, sys_type, force_group, ff_atom_types_indices)

                        # store in Molsys
                        self.reduced_indexed_ff_optimizable[sys_type][force_group].append(reduced_indexed_ff_opt)
                        self.reduced_ff_optimizable_values[sys_type][force_group].append(reduced_ff_opt_values) 

                    elif force_group == 'CustomNonbondedForce':

                        custom_selected_indexed_ff_opt = []
                        nbfixes = []

                        for atomtype in self.dupes[sys_type].items():

                            for i, param_line in enumerate(self.openmm_systems[sys_type].custom_nb_params):

                                if atomtype[0] == param_line[-1]: 

                                    custom_selected_indexed_ff_opt.append(np.hstack((param_line, i)))

                            for i, param_line in enumerate(self.openmm_systems[sys_type].nbfix):

                                if (atomtype[0] == param_line[-2] or atomtype[0] == param_line[-1]) == True:

                                    nbfixes.append(np.hstack((param_line, i))) 

                        nbfixes = np.array(nbfixes)

                        # find duplicate nbfixes
                        reduced_nbfix = []
                        nbfix_dupes = []
                        i = 0
                        while i < len(nbfixes):

                            try:

                                check = np.unique(nbfixes[i][:2].astype('float') - nbfixes[i+1][:2].astype('float'))

                                if check == 0:

                                    reduced_nbfix.append(nbfixes[i+1])
                                    nbfix_dupes.append([i, i+1])

                                else:

                                    reduced_nbfix.append(nbfixes[i])
                                    reduced_nbfix.append(nbfixes[i+1])

                                i += 2

                            except IndexError:

                                continue

                        reduced_nbfix = np.array(reduced_nbfix)
                        self.nbfix_dupes[sys_type] = nbfix_dupes

                        # duplicate removal from custom_selected_indexed_ff_opt
                        delete_list = []

                        for index, selected_parameters in enumerate(custom_selected_indexed_ff_opt):

                            if selected_parameters[-1] in self.to_be_removed[sys_type]:

                                delete_list.append(index)

                        custom_reduced_indexed_ff_opt = np.delete(custom_selected_indexed_ff_opt, delete_list, axis=0)
                        custom_reduced_indexed_ff_opt = custom_reduced_indexed_ff_opt[custom_reduced_indexed_ff_opt[:,-1].argsort()]
                        custom_reduced_indexed_ff_opt[:,0] = custom_reduced_indexed_ff_opt[:,0] * (2**(-1/6) * 2) # convert r/sigma to sigma
                        custom_reduced_indexed_ff_opt[:,1] = np.abs(custom_reduced_indexed_ff_opt[:,1]) # convert -epsilon to epsilon
                        custom_reduced_ff_opt_values = custom_reduced_indexed_ff_opt[:,:-2]
      
                        # store in Molsys
                        self.reduced_indexed_ff_optimizable[sys_type][force_group].append(custom_reduced_indexed_ff_opt)
                        self.reduced_ff_optimizable_values[sys_type][force_group].append(custom_reduced_ff_opt_values) 

                        if (isinstance(reduced_nbfix, np.ndarray) and len(reduced_nbfix) != 0) == True:
                        
                            # store in Molsys as 2nd array of force_group (index 1)
                            self.reduced_indexed_ff_optimizable[sys_type][force_group].append(reduced_nbfix)  
                            self.reduced_ff_optimizable_values[sys_type][force_group].append(reduced_nbfix[:,:-3])

            #sort dict keys bc python is stupid
            self.reduced_ff_optimizable_values[sys_type] = dict(sorted(self.reduced_ff_optimizable_values[sys_type].items()))  
            self.reduced_indexed_ff_optimizable[sys_type] = dict(sorted(self.reduced_indexed_ff_optimizable[sys_type].items()))           

        #check user input compatibility
        if list(sorted(self.reduced_indexed_ff_optimizable.keys())) == ['all', 'nosol']:

            for fg_nr, force_group in enumerate(self.reduced_indexed_ff_optimizable['all'].keys()):

                if force_group == 'NonbondedForce':

                    if not sorted(set(self.reduced_indexed_ff_optimizable['all'][force_group][0][:,1])) == sorted(set(self.reduced_indexed_ff_optimizable['nosol'][force_group][0][:,1])):

                        raise ValueError('Sliced atoms in NonbondedForce do not match. Check slice_list.')
                    
                elif force_group in ['NBException', 'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']: 

                    if not sorted(set(np.array(self.reduced_indexed_ff_optimizable['all'][force_group][0], dtype = object)[:,0])) == \
                            sorted(set(np.array(self.reduced_indexed_ff_optimizable['nosol'][force_group][0], dtype = object)[:,0])):

                        raise ValueError('Sliced atoms do not match. Check slice_list.')
                    
                elif force_group == 'CustomNonbondedForce':

                    if not sorted(set(self.reduced_indexed_ff_optimizable['all'][force_group][0][:,2])) == sorted(set(self.reduced_indexed_ff_optimizable['nosol'][force_group][0][:,2])):

                        raise ValueError('Sliced atoms in CustomNonbondedForce do not match. Check slice_list.')

        elif list(sorted(self.reduced_indexed_ff_optimizable.keys())) == ['all', 'mol1', 'mol2']:

            for fg_nr, force_group in enumerate(self.reduced_indexed_ff_optimizable['all'].keys()): 

                if force_group == 'NonbondedForce':

                    if not sorted(set(self.reduced_indexed_ff_optimizable['all'][force_group][0][:,1])) == sorted(set(np.concatenate((self.reduced_indexed_ff_optimizable['mol1'][force_group][0][:,1] \
                                                                                                  , self.reduced_indexed_ff_optimizable['mol2'][force_group][0][:,1])))):

                        raise ValueError('Sliced atoms in NonbondedForce do not match. Check slice_list.')
                    
                elif force_group in ['NBException', 'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']: 

                    if not sorted(set(np.array(self.reduced_indexed_ff_optimizable['all'][force_group][0], dtype = object)[:,0])) == \
                        sorted(set(np.concatenate((np.array(self.reduced_indexed_ff_optimizable['mol1'][force_group][0], dtype = object)[:,0] \
                                                    , np.array(self.reduced_indexed_ff_optimizable['mol2'][force_group][0], dtype = object)[:,0])))):

                        raise ValueError('Sliced atoms do not match. Check slice_list.')
                    
                elif force_group == 'CustomNonbondedForce':

                    if not sorted(set(self.reduced_indexed_ff_optimizable['all'][force_group][0][:,2])) == sorted(set(np.concatenate((self.reduced_indexed_ff_optimizable['mol1'][force_group][0][:,2] \
                                                                                                    , self.reduced_indexed_ff_optimizable['mol2'][force_group][0][:,2])))):
                        
                        raise ValueError('Sliced atoms in CustomNonbondedForce do not match. Check slice_list.')

                    
    def expand_reduced_parameters(self): #TODO patch the hybrid type
        """
        puts parameter values from reduced_ff_optimizable_values['all'] back into ff_optimizable[sys_type] (and custom_nb_params, nbfix if required)

        internal parameters:
            self.reduced_ff_optimizable_values['all']
            self.reduced_indexed_ff_optimizable[sys_type]
            self.dupes['all']
            self.nbfix_dupes
            self.interaction_dupes
            self.openmm_systems[sys_type].ff_optimizable
            (self.openmm_systems[sys_type].custom_nb_params)
            (self.openmm_systems[sys_type].nbfix)
            self.hybrid
            (self.hybrid_sys_types)

        sets:
            self.openmm_systems[sys_type].ff_optimizable
            self.openmm_systems[sys_type].custom_nb_params
            self.openmm_systems[sys_type].nbfix
        """

        for force_group in self.reduced_ff_optimizable_values['all'].keys():

            if force_group == 'NonbondedForce':

                for array_no, _array in enumerate(self.reduced_indexed_ff_optimizable['all'][force_group]):

                    for line_no, parameter_line in enumerate(_array):

                        for atom_type in self.dupes['all'].keys():

                            if atom_type == parameter_line[1]:

                                for atom_index in self.dupes['all'][atom_type]:

                                    if 'CustomNonbondedForce' in list(self.openmm_systems['all'].extracted_ff.keys()): #NBFIX

                                        reduced_ff_opt_values = self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no]
                                        reduced_ff_opt_values = np.append(reduced_ff_opt_values, [1.0, 0.0]) #reintroduce faux sigma & epsilon
                                        self.openmm_systems['all'].ff_optimizable[force_group][array_no][atom_index] = \
                                            tuple(reduced_ff_opt_values)

                                    else:
                                        self.openmm_systems['all'].ff_optimizable[force_group][array_no][atom_index] = \
                                            tuple(self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no])

                                for sys_type in list(self.reduced_ff_optimizable_values.keys())[1:]: # mol1&2 / nosol

                                    if atom_type in list(self.dupes[sys_type].keys()):

                                        for atom_idx in self.dupes[sys_type][atom_type]:

                                            if 'CustomNonbondedForce' in list(self.openmm_systems[sys_type].extracted_ff.keys()): #NBFIX

                                                reduced_ff_opt_values = self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no]
                                                reduced_ff_opt_values = np.append(reduced_ff_opt_values, [1.0, 0.0]) #reintroduce faux sigma & epsilon
                                                self.openmm_systems[sys_type].ff_optimizable[force_group][array_no][atom_index] = \
                                                    tuple(reduced_ff_opt_values)

                                            else:

                                                if self.hybrid == False:

                                                    self.openmm_systems[sys_type].ff_optimizable[force_group][array_no][atom_idx] = \
                                                        tuple(self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no]) 
                                                    
                                                elif self.hybrid == True:

                                                        self.openmm_systems[sys_type].ff_optimizable[force_group][array_no][atom_idx]['charge'] = \
                                                            self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no][0]


            elif force_group == 'CustomNonbondedForce': 

                # collect optimized values from arrays
                for array_no, _array in enumerate(self.reduced_indexed_ff_optimizable['all'][force_group]):

                    if _array.shape[1] == 4: # array at index 0 should be custom_nb_params

                        gmx_to_charmm_conversion_array = copy.deepcopy(self.reduced_ff_optimizable_values['all'][force_group][array_no])

                        gmx_to_charmm_conversion_array[:,0] = gmx_to_charmm_conversion_array[:,0] / (2**(-1/6) * 2)
                        gmx_to_charmm_conversion_array[:,1] = -1*gmx_to_charmm_conversion_array[:,1]

                        for line_no, parameter_line in enumerate(_array):

                            self.openmm_systems['all'].custom_nb_params[parameter_line[-1]][:2] = gmx_to_charmm_conversion_array[line_no]

                            for sys_type in list(self.reduced_ff_optimizable_values.keys())[1:]: # mol1&2 / nosol

                                if self.hybrid == False:

                                    for param_line_no, param_line in enumerate(self.openmm_systems[sys_type].custom_nb_params):

                                        if param_line[-1] == parameter_line[2]:

                                            self.openmm_systems[sys_type].custom_nb_params[param_line_no][:2] = gmx_to_charmm_conversion_array[line_no]

                                elif self.hybrid == True:

                                    if parameter_line[2] in list(self.dupes[sys_type].keys()):

                                        for atom_idx in self.dupes[sys_type][parameter_line[2]]:

                                            # careful static: 'CustomNonbondedForce' always has two arrays, here values are broadcasted from the array at index 0 to the 'NonbondedForce' array at index 0.    
                                            self.openmm_systems[sys_type].ff_optimizable['NonbondedForce'][0][atom_idx]['lj_sigma'] = self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no][0]
                                            self.openmm_systems[sys_type].ff_optimizable['NonbondedForce'][0][atom_idx]['lj_eps'] = self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no][1]

                    elif _array.shape[1] == 5: # array at index 1 should be NBFIX

                        for pair_type_idx in self.nbfix_dupes['all']: 

                            for line_no, line in enumerate(_array):

                                if line[-1] in pair_type_idx: # check if index matches

                                    for nbfix_index in pair_type_idx:

                                        self.openmm_systems['all'].nbfix[nbfix_index][:-2] = self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no]

                                for sys_type in list(self.reduced_ff_optimizable_values.keys())[1:]: # mol1&2 / nosol

                                    if self.hybrid == False:
                                        
                                        for nbfix_no, nbfix in enumerate(self.openmm_systems[sys_type].nbfix):

                                            if (np.all(line[2:4] == nbfix[2:4]) or np.all(line[2:4][::-1] == nbfix[2:4])) == True: # check if atomtypes match

                                                self.openmm_systems[sys_type].nbfix[nbfix_no][:2] = self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no]
                

                # calc acoef, bcoef
                self.calculate_acoef_bcoef()
                
           
            elif force_group in ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'NBException']: 

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
            self.reduced_ff_optimizable_values['all']

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
            self.reduced_ff_optimizable_values['all']
        """

        for force_group in self.vectorized_reduced_ff_optimizable_values.keys():

            for array_no, _array in enumerate(self.vectorized_reduced_ff_optimizable_values[force_group]):

                self.reduced_ff_optimizable_values['all'][force_group][array_no] = np.array(_array).reshape(np.array(self.reduced_ff_optimizable_values['all'][force_group][array_no]).shape)


    def merge_vectorized_parameters(self): 
        """
        'flattens' the dictionary of vectorized parameters of all selected force groups
        into one long vector

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
                slice_start = slice_end


    def scale_initial_parameters(self): 
        """
        Scales the magnitude of the parameters extracted from ff_optimizable using the z-score. This function 
        is run once before the parametrization loop to set constant scaling factors (self.scaling_factors).

        internal parameters:
            self.vectorized_parameters : np.array

        sets:
            self.scaling_factors, not mutable
            self.scaled_parameters, mutable
        """

        individual_mean = np.mean(self.vectorized_parameters)
        individual_sdev = np.std(self.vectorized_parameters)
        scaling_factors = (self.vectorized_parameters - individual_mean) / individual_sdev # TODO: keep these const or recalc??? 
        self.scaling_factors = scaling_factors.astype('float')
        scaled_parameters = self.vectorized_parameters * self.scaling_factors
        self.scaled_parameters = scaled_parameters.astype('float')


    def scale_parameters(self):
        """
        Scales the magnitude of the parameters extracted from ff_optimizable using the z-score. 

        internal parameters:
            self.vectorized_parameters : np.array
            self.scaling_factors : np.array

        sets:
            self.scaled_parameters, mutable
        """

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
        path += '/'

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

def read_qm_charges(read_lines, charge_type, path, n, cp2k_outfilename, ini_coords):

    """
    reads cp2k .out files and collects the calculated atom charges based on the type


    Parameters
    -----------
    read_lines : list of str
        read-in file content
    charge_type : str
        one of the following: 'Mulliken', 'Hirshfeld', 'RESP'     
    path : str
        path to externally cp2k optimized structures
    n : int
        frame number or conformation number
    cp2k_outfilename : str
        equal to <cp2k>.out
    ini_coords : np.array
        unoptimized coordinates of conformations

    returns: 
        charges (qm charges) as numpy arrary   
    """

    read = read_lines

    assert charge_type in ['Mulliken', 'Hirshfeld', 'RESP'], 'invalid charge_type {}'.format(charge_type)

    if charge_type == 'Mulliken':
        mullken_start = [index for index, string in enumerate(read) if 'Mulliken Population Analysis' in string]
        charges = np.loadtxt(path + '/frame' + str(n) + '/' + cp2k_outfilename + '.out',
                            skiprows = mullken_start[0] + 3,
                            max_rows = ini_coords.shape[1], usecols=4, dtype=float)

    elif charge_type == 'Hirshfeld':
        hirshfeld_start = [index for index, string in enumerate(read) if 'Hirshfeld Charges' in string]
        charges = np.loadtxt(path + '/frame' + str(n) + '/' + cp2k_outfilename + '.out',
                            skiprows = hirshfeld_start[0] + 3, max_rows = ini_coords.shape[1],
                            usecols=5, dtype=float)

    elif charge_type == 'RESP':
        resp_start = [index for index, string in enumerate(read) if 'RESP charges:' in string]
        charges = np.loadtxt(path + '/frame' + str(n) + '/' + cp2k_outfilename + '.out',
                            skiprows = resp_start[0] + 3,
                            max_rows = ini_coords.shape[1], usecols=3, dtype=float)
            
    return charges

def read_qm_energy_forces(read, n, cp2k_outfilename, path, ini_coords):

    """
    Useful if cp2k calculations have been run externally (e.g. on a cluster).
    reads cp2k .out files and extracts energy and forces from it. 

    Parameters
    ----------
    read : list of str
        read-in file content
    n : int
        frame number or conformation number
    cp2k_outfilename : str
        equal to <cp2k>.out
    path : str
        path to externally cp2k optimized structures
    ini_coords : np.array
        unoptimized coordinates of conformations
    
    returns:
        energy, forces : quantum energy & forces as float & np.array
    """

    energy_line = [index for index, string in enumerate(read) if
                'ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:'
                in string]

    if len(energy_line) == 0:
        raise ValueError('ENERGY not found in frame'+str(n)+'/'+cp2k_outfilename+'.out')

    energy = float(re.findall(r"[-+]?(?:\d*\.*\d+)", read[energy_line[-1]])[0])

    forces_start = [index for index, string in enumerate(read) if 'ATOMIC FORCES in [a.u.]' in string]

    forces = np.loadtxt(path + 'frame' + str(n) + '/' + cp2k_outfilename + '.out',
                            skiprows = forces_start[-1] + 3,
                            max_rows = ini_coords.shape[1], usecols=(3, 4, 5), dtype=float)
                            
    return energy, forces # in Hartree, Hartree/Bohr