#### package imports ####
import copy
import numpy as np
import Coord_Toolz.mdanalysis as ct
from .paths import *
import MDAnalysis as mda
import os, re
from ASE_interface.ase_calculation import *

#### define initial parameters of molecular system that go into the objective function ####

class Molecular_system:
    os.environ['OMP_NUM_THREADS'] = '6'
    """
    Contains properties of molecular system that is to be parametrized.

    Parameters
    ----------
    system_type : str
        Choice of ['1_gas_phase', '2_gas_phase', '1_solvent', '2_solvent']: the number denotes the amount of molecules
        to parametrize, the string specifies whether the molecule(s) is/are in gas phase or in solution (e.g. water).
    parametrization_type : str
        Choice of ('all_forces', 'net_forces'): The first takes all forces in the system as-is, the 2nd choice allows
        the computation of net forces (intramolecular forces of a molecule solved in water, intermolecular forces
        between two molecules, intermolecular forces between two solvated molecules)
    parametrization_methods : str
        Choice of ['energy', 'forces', 'energy&forces']: Which property is compared in the objective function

    paths : System.paths.Paths object
        Stores all relevant paths to files and folders

    ini_coords: numpy array of shape (n_conformations, n_atoms, 3)
        coordinates of the sampled structures in Angström
    ase_sys : ASE_interface.ase_calculation object
        ASE calculation instance
    openmm_sys : OMM_interface.openmm object
        OpenMM system instance
    n_conformations : int
        number of conformations aka sampled structures
    n_atoms : int
        number of atoms

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
    opt_...:
        optimized
    ..._mol1:
        belongs two moleecule1
    ..._mol2:
        belongs to molecule2
    weights: numpy array w/ len = n_conformations
        weights that are applied to the conformations
    """

    def __init__(self, system_type: str, parametrization_type: str, parametrization_method: str):
        """
        Initialize object with desired settings.
        """
        self.paths = None

        self.system_types = ['1_gas_phase', '2_gas_phase', '1_solvent', '2_solvent']
        self.parametrization_types = ('total_properties', 'net_properties')
        self.parametrization_methods = ['energy', 'forces', 'energy&forces']

        base_dict = {'all': None}
        base_dict_net2 = {'all': None,
                            'mol1': None,
                            'mol2': None}
        base_dict_solv1 = {'all': None,
                           'nosol': None}
        

        # read-in params
        self.system_type = system_type
        self.parametrization_type = parametrization_type
        self.parametrization_method = parametrization_method

        assert self.system_type in [systype for systype in self.system_types], "System of type {} is" \
                                                                                " not implemented.".format(system_type)
        assert self.parametrization_type in [ptype for ptype in self.parametrization_types], "Parametrization" \
                                                        " of type {} is not implemented.".format(parametrization_type)
        assert self.parametrization_method in [pmeth for pmeth in self.parametrization_methods], "Parametrization" \
                                                        " method {} is not implemented.".format(parametrization_method)
        
        # init data storage for different cases
        if self.parametrization_type == 'net properties':

            if self.system_type.find('2') != -1:

                self.ini_coords = copy.deepcopy(base_dict_net2)
                self.n_atoms = copy.deepcopy(base_dict_net2)
                self.ase_sys = copy.deepcopy(base_dict_net2)
                self.openmm_sys = copy.deepcopy(base_dict_net2)
                self.opt_coords = copy.deepcopy(base_dict_net2)
                self.mm_charges = copy.deepcopy(base_dict_net2)
                self.qm_charges = copy.deepcopy(base_dict_net2)

                if self.parametrization_method.find('energy') != -1:

                    self.mm_energies = copy.deepcopy(base_dict_net2)
                    self.qm_energies = copy.deepcopy(base_dict_net2)
                    self.eqm_bsse = copy.deepcopy(base_dict_net2)
                    
                if self.parametrization_method.find('forces') != -1:

                    self.mm_forces = copy.deepcopy(base_dict_net2)
                    self.qm_forces = copy.deepcopy(base_dict_net2)
            
                print("...Now set up the MDA_reader...")
                print("...define mol1 & mol2...")
                print("MDA_reader.mol1 = MDA_reader.delete_one_molecule('not resid 2')")
                print("MDA_reader.mol2 = MDA_reader.delete_one_molecule('not resid 1')")

            elif self.system_type.find('1_solvent'):

                self.ini_coords = copy.deepcopy(base_dict_solv1)
                self.n_atoms = copy.deepcopy(base_dict_solv1)
                self.ase_sys = copy.deepcopy(base_dict_solv1)
                self.openmm_sys = copy.deepcopy(base_dict_solv1)
                self.opt_coords = copy.deepcopy(base_dict_solv1)
                self.mm_charges = copy.deepcopy(base_dict_solv1)
                self.qm_charges = copy.deepcopy(base_dict_solv1)

                if self.parametrization_method.find('energy') != -1:

                    self.mm_energies = copy.deepcopy(base_dict_solv1)
                    self.qm_energies = copy.deepcopy(base_dict_solv1)
                    self.eqm_bsse = copy.deepcopy(base_dict_solv1)
                    
                if self.parametrization_method.find('forces') != -1:

                    self.mm_forces = copy.deepcopy(base_dict_solv1)
                    self.qm_forces = copy.deepcopy(base_dict_solv1)

                print('...Now set up the MDA_reader...')
                print('...define the molecule...')
                print('MDA_reader.nosol = MDA_reader.remove_water_ions()')

        else:

            self.ini_coords = copy.deepcopy(base_dict)
            self.n_atoms = copy.deepcopy(base_dict)
            self.ase_sys = copy.deepcopy(base_dict)
            self.openmm_sys = copy.deepcopy(base_dict)
            self.opt_coords = copy.deepcopy(base_dict)
            self.mm_charges = copy.deepcopy(base_dict)
            self.qm_charges = copy.deepcopy(base_dict)

            if self.parametrization_method.find('energy') != -1:

                self.mm_energies = copy.deepcopy(base_dict)
                self.qm_energies = copy.deepcopy(base_dict)
                self.eqm_bsse = copy.deepcopy(base_dict)
                
            if self.parametrization_method.find('forces') != -1:

                self.mm_forces = copy.deepcopy(base_dict)
                self.qm_forces = copy.deepcopy(base_dict)

        self.weights = None

    def set_ini_coords(self, MDA_reader_object):
        """
        reads in the initial atoms & coordinates from file(s) using the MDA_reader, stores them in Molecular_system,
        and acquires properties derived from them based on the settings in Molecular_system.  

        Parameters
        ----------
        MDA_reader_object: Coord_Toolz.mdanalysis.MDA_reader object
            Contains MDA Universe and atomgroups
        """

        coords = ct.get_coords(MDA_reader_object.all.atoms)
        self.n_conformations = len(coords) 
        self.n_atoms['all'] = len(MDA_reader_object.all.atoms)

        self.ini_coords['all'] = coords

        assert self.paths is not None, 'Molecular_system.paths not set'

        assert self.paths.mm_traj is not None, 'Cnformations trajectory not found.'

        assert self.paths.mm_crd is not None, 'Coordinate file not found.'

        assert self.paths.mm_top is not None, 'Topology fiel not found.'

        if MDA_reader_object.nosol is not None:

            nosol_coords = ct.get_coords(MDA_reader_object.nosol)
            self.ini_coords['nosol'] = nosol_coords
            self.n_atoms['nosol'] = len(MDA_reader_object.nosol)

            assert self.paths.mm_nosol_crd is not None, 'Coordinate file for molecule '\
                                                        'w/o solvent not found.'
            assert self.paths.mm_nosol_top is not None, 'Topology file for molecule '\
                                                        'w/o solvent not found.'
            
            print('...now set up the 2 OpenMM systems...')


        elif MDA_reader_object.mol1 is not None:

            mol1_coords = ct.get_coords(MDA_reader_object.mol1)
            self.ini_coords['mol1'] = mol1_coords
            self.n_atoms['mol1'] = len(MDA_reader_object.mol1)
            mol2_coords = ct.get_coords(MDA_reader_object.mol2)          
            self.ini_coords['mol2'] = mol2_coords
            self.n_atoms['mol2'] = len(MDA_reader_object.mol2)

            assert self.paths.mm_mol1_crd is not None, 'Coordinate file for molecule 1 '\
                                                        'w/o molecule 2 not found.'
            assert self.paths.mm_mol1_top is not None, 'Topology file for molecule '\
                                                        'w/o molecule 2 not found.'
            assert self.paths.mm_mol2_crd is not None, 'Cordinate file for molecule 2 '\
                                                        'w/o molecule 1 not found'
            assert self.paths.mm_mol2_top is not None, 'Topology file for molecule 2 '\
                                                        'w/o molecule 1 not found.'
            
            print('...now set up the 3 OpenMM systems...')

        else:

            print('...now set up the OpenMM system...')

    def generate_qm_energies_forcesraw_optcoords(self, run_type='single_point', coords=None):
        """
        calculates QM forces and energies of the conformations using ASE

        Parameters
        ----------
        run_type : str, default = 'single point'
            type of QM calculation: Choose btwn 'single point' (calc as-is), 'optimization' (geometry optimization),
            or 'gas_phase' (single-point w/o water)
        coords : np.array, default = None
            coordinates of all the conformations

        self.ini_coords
        self.ase_sys

        sets:
            self.qm_energies
            self.qm_forces
            or
            self.qm_energies_gp
            self.qm_forces_gp
            if 'gas_phase' is selected
            and optionally
            self.opt_coords
            if 'optimization' is selected
        """

        assert run_type in [rt for rt in ['single_point','optimization','gas_phase']], "Calculation of type {} is" \
                                                                                " not implemented.".format(run_type)

        if coords is None:
            assert self.ini_coords is not None, 'coordinates of the conformations are missing.'
            coords = self.ini_coords

        # initialize arrays
        # len(coords) is the number of conformations, coords.shape[1] is the number of atoms

        eqm = np.zeros((len(coords), 1))
        fqm = np.zeros((len(coords), coords.shape[1], 3))

        if run_type == 'optimization':
            opt_coords = np.zeros((len(coords), coords.shape[1], 3))

        # fill arrays
        for n in range(len(coords)):

            self.ase_sys.run_calculation(coords[n], run_type)

            eqm[n] = self.ase_sys.energies
            fqm[n, :, :] = self.ase_sys.forces
            if run_type == 'optimization':
                opt_coords[n, :, :] = self.ase_sys.opt_coords

        self.qm_energies = eqm
        self.qm_forces = fqm

        if run_type == 'optimization':
            self.opt_coords = opt_coords

        elif run_type == 'gas_phase':

            self.qm_energies_gp = eqm
            self.qm_forces_gp = fqm

    def generate_qm_charges_energies_forcesraw_optcoords(self, paths, cp2k_input: str, omp_threads: int,
                                                      cp2k_binary_name: str, charge_type: str):

        """
        Calculates QM energies, forces, and charges using native cp2k and also the corresponding MM charges

        Parameters
        ----------
        paths : paths.Paths object
            contains paths and filenames etc
        cp2k_input : str
            all the commands and strings that go into a cp2k .inp file (see cp2k doc). It is recommended to load the
            input into a separate variable beforehand
        omp_threads : int
            number of openMP threads to use for the parallel cp2k calculation
        cp2k_binary_name : str
            name by which the cp2k binary is called on your machine
        charge_type : str
            Options are 'Mulliken', 'Hirshfeld', 'RESP'

        self.n_conformations : number of conformations
        self.naked_molecule_n_atoms : number of atoms in molecule w/o water
        self.openmm_sys : OpenMM system object

        sets:
            self.mm_charges_gp : classical charges of the molecule w/o water
            self.qm_energies_gp : quantum energies of the molecule w/o water
            self.qm_forces_gp : "-" forces "-"
            self.opt_charges : "-" charges "-"
        """

        #### construct charge calculator object ####
        from ..Direct_cp2k_calculation.direct_cp2k import Direct_Calculator as cc
        run_type = 'charges'
        charge_calc = cc(paths, run_type)

        #### initialize arrays (for unsolvated molecule) ####
        cmm_gp = np.zeros((self.n_conformations, self.naked_molecule_n_atoms))

        eqm_gp = np.zeros((self.n_conformations, 1))

        fqm_gp = np.zeros((self.n_conformations, self.naked_molecule_n_atoms, 3))

        cqm_gp = np.zeros((self.n_conformations, self.naked_molecule_n_atoms))

        #### fill the arrays ####
        for n in range(self.n_conformations):

            charge_calc.generate_cp2k_input_file(cp2k_input)
            charge_calc.run_cp2k(omp_threads, cp2k_binary_name)
            charge_calc.extract_charges(charge_type, self.naked_molecule_n_atoms)

            cqm_gp[n, :] = charge_calc.charges
            eqm_gp[n] = charge_calc.energy * 2.62549961709828E+03 #Hartree to kJ/mol

            fqm_gp[n, :, :] = charge_calc.forces * 2.62549961709828E+03 * 10.0 #a.u. to kJ/mol/nm

            cmm_gp[n, :] = self.openmm_sys.charges

        self.mm_charges_gp = cmm_gp

        self.qm_energies_gp = eqm_gp

        self.qm_forces_gp = fqm_gp

        self.opt_charges = cqm_gp

    def read_qm_charges_energies_forcesraw_optcoords(self, path: str, filename: str, outfilename: str, charge_type: str):

        """
        Useful if cp2k calculations have been run externally (e.g. on a cluster).
        reads cp2k .out files and extracts charges, energies, and forces from it

        Parameters
        ----------
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
        self.naked_molecule_n_atoms : number of atoms in molecule w/o water

        sets:
            self.opt_coords : geometry-optimized coordinates of conformations
            self.qm_forces : quantum forces
            self.qm_energies : quantum energies
            self.qm_charges : quantum charges
        """

        if path[-1] != '/':
            path = path[:-1]

        #### init arrays ####
        self.opt_coords = np.zeros(self.ini_coords.shape)
        self.qm_forces = np.zeros((len(self.ini_coords), self.ini_coords.shape[1], 3))
        self.qm_energies = np.zeros((len(self.ini_coords), 1))
        self.qm_charges = np.zeros((len(self.ini_coords), self.ini_coords.shape[1]))

        #### fill the arrays ####

        for n in range(self.n_conformations):

            assert charge_type in ['Mulliken', 'Hirshfeld', 'RESP'], 'invalid charge_type {}'.format(charge_type)
            assert (type(self.naked_molecule_n_atoms) == int and self.naked_molecule_n_atoms > 0) == True, \
                'invalid number of atoms {}'.format(self.naked_molecule_n_atoms)

            f = open(path + outfilename + '.out', 'r')
            read = []
            for index, line in enumerate(f.readlines()):
                line = line.strip()
                read.append(line)
            f.close()

            energy_line = [index for index, string in enumerate(read) if
                           'ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:'
                           in string]
            energy = float(re.findall(r"[-+]?(?:\d*\.*\d+)", read[energy_line[-1]])[0])

            if charge_type == 'Mulliken':
                mullken_start = [index for index, string in enumerate(read) if 'Mulliken Population Analysis' in string]
                charges = np.loadtxt(path + '/frame' + str(n) + '/' + outfilename + '.out',
                                     skiprows=mullken_start[0] + 3,
                                     max_rows=self.naked_molecule_n_atoms, usecols=4, dtype=float)

            elif charge_type == 'Hirshfeld':
                hirshfeld_start = [index for index, string in enumerate(read) if 'Hirshfeld Charges' in string]
                charges = np.loadtxt(path + '/frame' + str(n) + '/' + outfilename + '.out',
                                     skiprows=hirshfeld_start[0] + 3, max_rows=self.naked_molecule_n_atoms,
                                     usecols=5, dtype=float)

            elif charge_type == 'RESP':
                resp_start = [index for index, string in enumerate(read) if 'RESP charges:' in string]
                charges = np.loadtxt(path + '/frame' + str(n) + '/' + outfilename + '.out',
                                     skiprows=resp_start[0] + 3,
                                     max_rows=self.naked_molecule_n_atoms, usecols=3, dtype=float)

            forces_start = [index for index, string in enumerate(read) if 'ATOMIC FORCES in [a.u.]' in string]
            forces = np.loadtxt(path + '/frame' + str(n) + '/' + outfilename + '.out',
                                     skiprows=forces_start[0] + 3,
                                     max_rows=self.naked_molecule_n_atoms, usecols=(3, 4, 5), dtype=float)

            self.qm_forces[n, :, :] = forces
            self.qm_energies[n] = energy
            self.qm_charges[n, :] = charges

            u = mda.Universe(path + '/frame' + str(n) + '/' + filename + '_opt-pos-1.xyz')
            coords = ct.get_coords(u.atoms)[-1]
            self.opt_coords[n, :, :] = coords

    def generate_qm_energies_forces(self, atomgroup, atomgroup_name, paths, cp2k_inp):

        """
        calculates QM energies and forces needed for net forces (raw sys forces - mol1 with water forces -
        mol2 with water forces = net forces)

        Parameters
        ----------
        atomgroup : MDAnalysis AtomGroup object
            e.g. molecule1
        atomgroup_name : str
            e.g. 'molecule1', needed for output files
        paths : paths.paths object
            contains working dir n stuff
        cp2k_inp : str
            input for the inp parameter of the ASE cp2k calculator w/ all the necessary cp2k control parameters

        returns:
            fqm : quantum forces of atomgroup
            eqm : quantum energies of atomgroup
        """

        os.chdir(paths.working_dir + paths.project_name)

        coords = ct.get_coords(atomgroup)

        for frame_nr, frame in enumerate(coords):
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
                        print_level='MEDIUM')
            CP2K.command = 'env OMP_NUM_THREADS=6 cp2k_shell.ssmp' # TODO: set OMP_NUM_THREADS somewhere else?
            ase_sys.atoms.calc = calc
            ase_sys.run_calculation(run_type='single_point')
            np.savetxt('forces_energy_' + atomgroup_name + '_frame' + str(frame_nr) + '.txt', ase_sys.forces,
                       header='E: ' + str(ase_sys.energy))
            outstr = 'cp cp2k.out ' + atomgroup_name + '_frame' + str(frame_nr) + '.out'
            os.system(outstr)
            os.system('rm cp2k.out')
            os.system('pkill cp2k_shell.ssmp')
            os.system('touch cp2k.out')

        #### init arrays ####
        fqm = np.zeros((len(coords), coords.shape[1], 3))
        eqm = np.zeros((len(coords), 1))

        #### fill arrays ####
        for n, frane in enumerate(coords):
            fqm[n, :, :] = np.genfromtxt('forces_energy_' + atomgroup_name + '_frame' + str(n) + '.txt', skip_header=1)
            f = open(
                'forces_energy_' + atomgroup_name + '_frame' + str(n) + '.txt', 'r')
            read = []
            for i, line in enumerate(f.readlines()):
                line = line.strip('# E:\n')
                if i == 0:
                    read.append(line)
            f.close()
            eqm[n] = float(read[0])

        return fqm, eqm

    def calculate_qm_net_forces(self, molnames, paths, cp2k_input):

        """
        Computes the net forces between 2 molecules by subtracting the forces caused by interaction w/ surrounding
        water. (raw sys forces - mol1 with water forces - mol2 with water forces = net forces)

        Parameters
        ----------
        molnames : dict
            names of molecules whose interaction forces w/ water have to be subtracted and their atom groups:
            'mol1' : mol1
        paths : paths.paths object
            contains working dir n stuff
        cp2k_inp : str
            input for the inp parameter of the ASE cp2k calculator w/ all the necessary cp2k control parameters

        """

        for molname in molnames.keys(): # aaaahhhhhh need another dict to store shit :(
            self.generate_qm_energies_forces(atomgroup, molname, paths, cp2k_input)
            net_forces_results[molname] = {
            'net_forces': net_forces,
            'qm_energies': self.qm_energies,  
            'qm_forces': self.qm_forces  # not sure if I missed any other results here
            }

            self.generate_qm_energies_forces(molnames[molname], molname, paths, cp2k_input)
            self.generate_qm_charges_energies_forcesraw_optcoords(molname, paths, cp2k_inp, omp_threads, cp2k_binary, charge_type)
            for mdanalysis in paths:
                return


        #TODO: make sense of this shit (in the above for loop?)
    def extract_ff_parameters(self):
        """
        Extract force field parameters from the ff_optimizable attribute.

        Returns
        -------
        ff_parameters : dict
            A dictionary containing force field parameters.
        """
        ff_parameters = {'NonbondedForce': {'charge': None}}  # can add more force field terms if there are more

        if 'NonbondedForce' in self.ff_optimizable:
            ff_parameters['NonbondedForce']['charge'] = copy.deepcopy(self.ff_optimizable['NonbondedForce'][0]['charge'])
        
        return ff_parameters

    def generate_mm_energies_forcesraw(self, coords=None):
        """
        collects the OpenMM system's classical properties (energies&forces)

        Parameters
        ----------
        coords : np.array, default = None
            coordinates of the conformations

        self.ini_coords : np.array, initial unoptimized coordinates of the conformations
        self.openmm_sys : OpenMM system object

        sets:
            self.mm_energies : classical energies
            self.mm_forces : classical forces
        """

        if coords is None:
            assert self.ini_coords is not None, 'coordinates of the conformations are missing.'
            coords = self.ini_coords

        # initialize arrays
        # len(coords) is the number of conformations, coords.shape[1] is the number of atoms
        emm = np.zeros((len(coords), 1))
        fmm = np.zeros((len(coords), coords.shape[1], 3))

        # fill arrays
        for n in range(len(coords)):

            self.openmm_sys.run_calculation(coords[n])

            emm[n] = self.openmm_sys.energies
            fmm[n, :, :] = self.openmm_sys.forces

        self.mm_energies = emm
        self.mm_forces = fmm


    def get_bsse(self, paths, cp2k_input: str, omp_threads: int, cp2k_binary_name: str):

        """
        calculate the basis set superposition error (BSSE) using native cp2k

        Parameters
        ----------
        project_path : str
            path to directory where the cp2k files will be read/written
        project_name : str
            name of the cp2k project, also used to name files
        cp2k_input : str
            all the commands and strings that go into a cp2k .inp file (see cp2k doc). It is recommended to load the
            input into a separate variable beforehand
        omp_threads : int
            number of openMP threads to use for the parallel cp2k calculation
        cp2k_binary_name : str
            name by which the cp2k binary is called on your machine

        sets:
            self.eqm_bsse : float, basis set superposition error
        """

        #### construct charge calculator object ####
        from ..Direct_cp2k_calculation.direct_cp2k import Direct_Calculator as cc
        run_type = 'BSSE'
        bsse_calc = cc(paths, run_type)

        #### run calc ####
        bsse_calc.generate_cp2k_input_file(cp2k_input)
        bsse_calc.run_cp2k(omp_threads, cp2k_binary_name)

        self.eqm_bsse = bsse_calc.bsse_total * 2.62549961709828E+03  # Hartree to kJ/mol

    # def get_dihedrals(self):

    def correct_charges(self, n_atoms: int, real_charge: int):
        """
        Use if the numerical total charge does not equal the real total charge (due to numerical errors)

        Parameters
        ----------
        n_atoms : int
            number of atoms of the molecule that should be parametrized
        real_charge : int
            charge of molecule that is to be parametrized

        self.ff_optimizable : dict containing optimized force field parameters

        """

        if len(self.ff_optimizable['NonbondedForce'][0]) > 0:

            if 'charge' in self.ff_optimizable['NonbondedForce'][0].dtype.names:

                total_charge = sum(self.ff_optimizable['NonbondedForce'][0]['charge'][:n_atoms])
                print('total charge = ' + str(total_charge))

                if total_charge != real_charge:

                    charge_correction = np.abs(total_charge - real_charge) / n_atoms
                    print('charge correction per atom = ' + str(charge_correction))

                    for atom in range(n_atoms):
                        self.ff_optimizable['NonbondedForce'][0]['charge'][atom] -= charge_correction

                    corrected_total_charge = sum(self.ff_optimizable['NonbondedForce'][0]['charge'][:n_atoms])

                    print('corrected total charge = ' + str(corrected_total_charge))

            else:
                raise NameError("'charge' not in ff_optimizable")

        else:
            raise KeyError("ff_optimizable has no key 'NonbondedForce'")


    def generate_weights(self):
        """
        For now weighs all conformations equally. Can be modified tho.

        sets:
            self.weights
        """

        self.weights = np.ones((self.n_conformations))
        self.weights = self.weights / np.sum(self.weights)





