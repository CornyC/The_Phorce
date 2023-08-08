#### package imports ####
import copy
import numpy as np

#### define initial parameters of molecular system that go into the objective function ####

class Molecular_system:
    """
    Contains properties of molecular system that is to be parametrized.

    Parameters
    ----------
    n_conformations : int
        number of conformations aka sampled structures
    n_atoms : int
        number of atoms
    ase_sys : ASE_interface.ase_calculation object
        ASE calculation instance
    openmm_sys : OMM_interface.openmm object
        OpenMM system instance
    naked_molecule : Coord_Toolz.mdanalysis.MDA_reader.molecule_only object
        water and ions removed from .pdb for gas phase/charge calculation
    molecule1 : Coord_Toolz.mdanalysis.MDA_reader.molecule1 object
        if interaction forces between two molecules are supposed to be optimized, this is molecule 1
    molecule2 : Coord_Toolz.mdanalysis.MDA_reader.molecule2 object
        if interaction forces between two molecules are supposed to be optimized, this is molecule 2
    naked_molecule_n_atoms : int
        number of atoms of molecule to be parametrized
    ini_coords: numpy array of shape (n_conformations, n_atoms, 3)
        coordinates of the sampled structures in Angström
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
    ..._gp:
        gas phase (no water)
    opt_...:
        optimized
    """

    def __init__(self, ini_coords, ase_sys, openmm_sys, naked_molecule=None, molecule1=None, molecule2=None):

        # read-in params
        self.ini_coords = ini_coords
        self.n_conformations = len(self.ini_coords)
        self.n_atoms = self.ini_coords.shape[1]
        self.ase_sys = ase_sys
        self.openmm_sys = openmm_sys
        self.naked_molecule = naked_molecule
        self.molecule1 = molecule1
        self.molecule2 = molecule2

        ### other system params ###
        self.naked_molecule_n_atoms = None
        self.weights = None

        # from md/ff #
        self.mm_energies = None
        self.mm_energies_gp = None
        self.mm_forces = None
        self.mm_forces_gp = None
        self.mm_charges = None
        self.mm_charges_gp = None
        self.ini_dihed = None

        # from dft #

        self.qm_energies = None
        self.qm_energies_gp = None
        self.qm_forces = None
        self.qm_forces_gp = None
        self.qm_charges = None
        self.qm_charges_gp = None
        self.eqm_bsse = None

        self.opt_coords = None
        self.opt_charges = None
        self.opt_dihed = None



    def get_qm_energies_forces_optcoords(self, run_type='single_point', coords=None):

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

    def get_mm_energies_forces(self, coords=None):

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


    def generate_weights(self):

        self.weights = np.ones((self.n_conformations))
        self.weights = self.weights / np.sum(self.weights)

    def get_charges(self, project_path: str, project_name: str, cp2k_input: str, omp_threads: int,
                    cp2k_binary_name: str, charge_type: str):

        """
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
        charge_type : str
            Options are 'Mulliken', 'Hirshfeld', 'RESP'
        """

        #### construct charge calculator object ####
        from ..Direct_cp2k_calculation.direct_cp2k import Direct_Calculator as cc
        run_type = 'charges'
        charge_calc = cc(project_path, project_name, run_type)

        #### initialize arrays (for unsolvated molecule) ####
        cmm_gp = np.zeros((self.n_conformations, self.naked_molecule_n_atoms))

        eqm_gp = np.zeros((self.n_conformations, 1))

        fqm_gp = np.zeros((self.n_conformations, self.naked_molecule_n_atoms, 3))

        cqm_gp = np.zeros((self.n_conformations, self.naked_molecule_n_atoms)) # which shape should this have?

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

    def get_bsse(self, project_path: str, project_name: str, cp2k_input: str, omp_threads: int, cp2k_binary_name: str,):

        """
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
        """

        #### construct charge calculator object ####
        from ..Direct_cp2k_calculation.direct_cp2k import Direct_Calculator as cc
        run_type = 'BSSE'
        bsse_calc = cc(project_path, project_name, run_type)

        #### run calc ####
        bsse_calc.generate_cp2k_input_file(cp2k_input)
        bsse_calc.run_cp2k(omp_threads, cp2k_binary_name)

        self.eqm_bsse = bsse_calc.bsse_total * 2.62549961709828E+03  # Hartree to kJ/mol

    #def get_dihedrals(self):


    def correct_charges(self, n_atoms: int, real_charge: int):
        """
        Use if the numerical total charge does not equal the real total charge (due to numerical errors)

        Parameters
        ----------
        n_atoms : int
            number of atoms of the molecule that should be parametrized
        real_charge : int
            charge of molecule that is to be parametrized
        """

        if len(self.ff_optimizable['NonbondedForce'][0]) > 0:
        
            if 'charge' in self.ff_optimizable['NonbondedForce'][0].dtype.names:
            
                total_charge = sum(self.ff_optimizable['NonbondedForce'][0]['charge'][:n_atoms])
                print('total charge = '+str(total_charge))

                if total_charge != real_charge:
                    
                    charge_correction = np.abs(total_charge - real_charge) / n_atoms
                    print('charge correction per atom = '+str(charge_correction))
                
                    for atom in range(n_atoms):
                        self.ff_optimizable['NonbondedForce'][0]['charge'][atom] -= charge_correction
                    
                    corrected_total_charge = sum(self.ff_optimizable['NonbondedForce'][0]['charge'][:n_atoms])

                    print('corrected total charge = ' + str(corrected_total_charge))

            else:
                raise NameError("'charge' not in ff_optimizable")
        
        else: 
            raise KeyError("ff_optimizable has no key 'NonbondedForce'")
