#### package imports ####

import numpy as np
import ase
from ase.optimize import LBFGS
from ase.calculators.cp2k import CP2K
from ase.constraints import FixInternals

#### run the QM calculation ####

class ASE_system:
    """
    Prepares and runs the QM calculation.

    Parameters
    ----------
    atom_symbols: ASE atoms.symbols object
        contains short identifiers for atom types
    positions: numpy array
        coordinates of one sampled structure
    cell_dimensions: numpy array of shape (3,)
        length of 3 cell edges in Angström
    pbc: boolean numpy array of shape (3,)
        Enabled/disabled periodic boundary conditions in x,y,z direction
    dihedral_freeze : list of list of int, default=None
        List of lists of wherein each inner list should contain 4 integers defining a torsion to be kept fixed.
    ase_constraints : list of ASE constraints, default=None
        List of ASE constraints to be applied during the scans.
        More information: https://wiki.fysik.dtu.dk/ase/ase/constraints.html
    shake_threshold : float
        Threshold of the SHAKE algorithm (constrains water vibrations)
    optimizer : any ase.optimizer, default=:obj:`ase.optimize.lbfgs.LBFGS*`
        ASE optimizer instance. For more info see: https://wiki.fysik.dtu.dk/ase/ase/optimize.html
    opt_fmax : float, default=1e-2
        The convergence criterion to stop the optimization is that the force on all individual atoms
        should be less than `fmax`.
    opt_log_file : str, default="-"
        File where optimization log will be stored. Use '-' for stdout.
    opt_traj_prefix : str, default="traj_"
        Prefix given to the pickle file used to store the trajectory during optimization.

    atoms : ASE atoms object
        contains symbols, positions, cell, pbc, calculator (see ASE doc)
    energy : float
        single-point energy of molecule in water obtained from DFT calculation in kJ/mol
    forces : numpy array
        forces an each atom as obtained from DFT calculation in kJ/mol/nm
    opt_coords : numpy array
        positions of atoms after DFT geometry optimization in nm
    energy_gp :
        single-point energy of molecule in gas phase obtained from DFT calculation in kJ/mol
    forces_gp :
        forces an each atom of gas-phase molecule as obtained from DFT calculation in kJ/mol/nm
    """

    def __init__(self, symbols=None, positions=None,  cell_dimensions=None, pbc=None, calculator=None,
                 dihedral_freeze=None, ase_constraints=None, shake_threshold=1e-7, optimizer=LBFGS, opt_fmax=1e-2,
                 opt_log_file="-", opt_traj_prefix="traj_"):

        self.symbols = symbols
        self.positions = positions
        self.cell = cell_dimensions
        self.pbc = pbc

        if self.pbc != None:
            assert self.cell != None, 'Periodic boundary conditions set but no cell dimensions given.'

        self.calculator = calculator
        self.dihedral_freeze = dihedral_freeze
        self.ase_constraints = ase_constraints
        self._shake_threshold = shake_threshold

        # ASE optimizer variables
        self._optimizer = optimizer
        self._opt_fmax = opt_fmax
        self._opt_logfile = opt_log_file
        self._opt_traj_prefix = opt_traj_prefix

        # data params
        self.atoms = None
        self.energy = None
        self.forces = None
        self.opt_coords = None
        self.energy_gp = None
        self.forces_gp = None

    def construct_atoms_object(self):

        atoms = ase.Atoms(self.symbols, self.positions)
        atoms.set_cell(self.cell)
        atoms.set_pbc(self.pbc)
        atoms.set_calculator(self.calculator)
        atoms.center()

        self.atoms = atoms

    def run_calculation(self, run_type='single_point'):

        if run_type == 'single_point':

            print('#############################################################')
            print('#           Running Single Point Calculation                #')
            print('#############################################################')

            self.energy = self.atoms.get_potential_energy() * 96.48530749925794  # eV to kJ/mol
            self.forces = self.atoms.get_forces() * 96.48530749925794 * 10.0  # eV/A to kJ/mol/nm

            print('#############################################################')
            print('#           Single Point Calculation successful             #')
            print('#############################################################')

        elif run_type == 'optimization':

            print('#############################################################')
            print('#             Running Geometry Optimization                 #')
            print('#############################################################')
            from ase.io import read, write

            constraints_list=[]

            if self.dihedral_freeze is not None:
                dihedrals_to_fix = []
                for dihedral in self.dihedral_freeze:
                    dihedrals_to_fix.append([self.atoms.get_dihedral(*dihedral) * np.pi / 180.0, dihedral])

                constraint = FixInternals(bonds=[], angles=[], dihedrals=dihedrals_to_fix,
                                          epsilon=self._shake_threshold)
                constraints_list.append(constraint)

            # Apply any ASE constraints
            # More information: https://wiki.fysik.dtu.dk/ase/ase/constraints.html
            if self.ase_constraints is not None:
                for constraint in self.ase_constraints:
                    constraints_list.append(constraint)

            if len(constraints_list) > 0:
                self.atoms.set_constraint(constraints_list)

            opt = self._optimizer(self.atoms, trajectory=self._opt_traj_prefix+".traj", logfile=self._opt_logfile)
            opt.run(fmax=self._opt_fmax)

            if self.dihedral_freeze is not None:
                del self.atoms.constraints

            # Get data

            self.energy = self.atoms.get_potential_energy() * 96.48530749925794  # eV to kJ/mol
            self.forces = self.atoms.get_forces() * 96.48530749925794 * 10.0 # eV/A to kJ/mol/nm (what OMM uses)
            self.opt_coords = self.atoms.get_positions() * 0.1 # Angstrom to nm


            print('#############################################################')
            print('#           Geometry Optimization successful                #')
            print('#############################################################')

        elif run_type == 'gas_phase':

            print('#############################################################')
            print('#           Running Single Point Calculation                #')
            print('#                      in gas phase                         #')
            print('#############################################################')

            assert (self.atoms.pbc == ([False, False, False]) and len(self.atoms.cell) == 0) is True, 'Remove pbc, ' \
                                                                                                      'cell, and H2O ' \
                                                                                                      'to run a gas ' \
                                                                                                      'phase calc.'

            self.energy_gp = self.atoms.get_potential_energy() * 96.48530749925794  # eV to kJ/mol
            self.forces_gp = self.atoms.get_forces() * 96.48530749925794 * 10.0  # eV/A to kJ/mol/nm

            print('#############################################################')
            print('#           Single Point Calculation successful             #')
            print('#############################################################')

        """
        elif run_type == 'charges': # does not work bc not implemented in ASE, see module Direct_cp2k_calculation instead

            print('#############################################################')
            print('#           Running Single Point Calculation                #')
            print('#                      in gas phase                         #')
            print('#                 and calculating RESP                      #')
            print('#############################################################')

            assert (self.atoms.pbc is ([False, False, False]) and len(self.atoms.cell) is 0) is True, 'Remove pbc, ' \
                                                                                                      'cell, and H2O ' \
                                                                                                      'to run a gas ' \
                                                                                                      'phase calc.'

            self.energy_gp = self.atoms.get_potential_energy() * 96.48530749925794  # eV to kJ/mol
            self.forces_gp = self.atoms.get_forces() * 96.48530749925794 * 10.0  # eV/A to kJ/mol/nm
            #self.charges = self.atoms.get_charges() #not implemented in ASE -.-

            print('#############################################################')
            print('#           Single Point Calculation successful             #')
            print('#############################################################')
        """
