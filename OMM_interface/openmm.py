#### package imports ####

import numpy as np
import openmm.unit as unit
import openmm
import openmm.app as app
from openmm.app import forcefield as ff
import copy, os
import fnmatch
from pathlib import Path
from System.paths import *
from math import sqrt


#### Setting up the OpenMM System ####

class OpenMM_system:
    """
    Creates the OpenMM System
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.System.html?highlight=getforces#openmm.openmm.System

    Parameters
    ----------
    topology_format : str, optional, default=None
        Available options are "AMBER", "GROMACS", "CHARMM", or "XML".
    top_file : str, optional, default=None
        Path to the AMBER, GROMACS or CHARMM topology file.
    topology : OpenMM topology object, optional
        Set OpenMM topology manually
    crd_format : str
        Available options are "AMBER", "GROMACS", "CHARMM", or "PDB".
    crd_file : str, optional, default=None
        Path to the AMBER, GROMACS or CHARMM coordinates file.
    charmm_param_file : str
        Path to the CHARMM param file (CHARMM stream file with .str extension). 
        Register the main CHARMM param stream file for your molecule here and if needed 
        have it read in other .str/.rtf files ;-)
    xml_files : list of str, optional, default=None
        Path to the .xml OpenMM topology files.
    pbc : bool
        Whether periodic boundary conditions are to be used or not
    cell : list of length 3
        cell edge lengths in nm
    angles : list of length 3
        cell angles in degree
    platform_name : str, default='CUDA'
        name of the platform; either 'CUDA','OpenCL','CPU'
    platform_properties : dict
        CUDA needs Precision and DeviceIndex to be set
        OpenCL needs Precision, DeviveIndex, and OpenCLPlatformIndex to be set
        CPU needs Precision and DeviceIndex
    integrator : openmm object, default=openmm.LangevinIntegrator
        MD integrator
    integrator_params : dict
        Keyword arguments passed to the integrator
    create_system_params : dict
        Keyword arguments passed to the top.createSystem instance

    other (internal) parameters :
        self.top : openmm object
            either AmberPrmtopFile, GromacsTopFile, CharmmPsfFile, or openmm native ForceField
        self.top_params : openmm object
            CharmmParameterSet
        self.crd : openmm object
            one of AmberInpcrdFile, GromacsGroFile, CharmmCrdFile, PDBFile
        self.platform : openmm.openmm.Platform
        self.system : openmm.openmm.System
        self.context : openmm.openmm.Context
        self.force_groups : dict
            keys see self.force_groups_dict, values are lists of openmm's force group indices
        self.extracted_ff : dict
            keys see self.force_groups, values are lists of np.recarrays containing the force field params
        self.ff_optimizable : dict
            contains force groups and their parameters that should be optimized
        self.custom_nb_params : np.array
            triggered if NBFIX present. Contains rmin(=r/sigma; nm), epsilon (kJ/mol)(in GMX/OMM units), and atom type (str)
        self.nbfix : np.array
            triggered if NBFIX is present. Contains rij (nm) and wdij (kJ/mol) (GMX/OMM units), 
            atom type 1, atom type 2 (str), and index 
        self.force_groups_dict : dict
                                 {'HarmonicBondForce': [],
                                  'HarmonicAngleForce': [],
                                  'PeriodicTorsionForce': [],
                                  'NonbondedForce': [],
                                  'CustomBondForce': [],
                                  'CustomAngleForce': [],
                                  'CustomTorsionForce': []
                                  'CustomNonbondedForce': [],
                                  "CMAPTorsionForce": [],
                                  "NBException": [],
                                  "CMMotionremover": []}
        self.charges : np.array
            mm charges of all atoms in the openmm system
    """


    def __init__(self, topology_format=None, top_file=None, topology=None, crd_format=None, crd_file=None,
                 charmm_param_file=None, xml_files=None, pbc=False, cell=None, angles=None):

        # Files to read in
        self.topology_format = topology_format
        self.top_file = top_file
        self.topology = topology
        self.crd_format = crd_format
        self.crd_file = crd_file
        self.charmm_param_file = charmm_param_file
        self.xml_files = xml_files

        # Box definitions (x, y, z are nanometers; the angles are degrees)
        self.pbc = pbc
        self.cell = np.asarray(cell, dtype=float)
        self.angles = np.asarray(angles, dtype=float)

        # OpenMM essential object instances
        self.platform_name = 'CUDA'
        self.platform_properties = {'Precision': 'single', 'DeviceIndex': '0'}
        self.integrator = openmm.LangevinIntegrator
        self.integrator_params = {'temperature': 310.0 * unit.kelvin, 'stepSize': 0.001 * unit.picoseconds,
                                  'frictionCoeff': 2.0 / unit.picoseconds}
        self.create_system_params = {'nonbondedMethod': app.PME, 'nonbondedCutoff': 1.2 * unit.nanometer,
                                         'switchDistance': 0.7 * unit.nanometer, 'constraints': app.HBonds,
                                         'rigidWater': False, 'temperature': 310.0 * unit.kelvin}

        # other params (required to create sys)
        self.top = None
        self.top_params = None
        self.crd = None
        self.platform = None
        self.system = None
        self.context = None

        # other params (required to extract FF)
        self.force_groups = {}
        self.extracted_ff = {}
        self.ff_optimizable = {}
        self.force_groups_dict = {'HarmonicBondForce': [],
                                  'HarmonicAngleForce': [],
                                  'PeriodicTorsionForce': [],
                                  'NonbondedForce': [],
                                  'CustomBondForce': [],
                                  'CustomAngleForce': [],
                                  'CustomTorsionForce': [],
                                  'CustomNonbondedForce': [],
                                  "CMAPTorsionForce": [],
                                  "NBException": [],
                                  "CMMotionremover": []}

        # data params
        self.charges = None

    def import_molecular_system(self):
        """
        reads in coords and topology

        Parameters
        ----------
        self.topology_format : str
            type of topology file, each MD software has its own :(
        self.top_file : str
            path to the topol file
        self.pbc : bool
            whether to use periodic boundary conditions
        self.crd_format : str
            .pdb is kind of a standard, nonetheless GROMACS also has .gro and CHARMM and AMBER have .crd

        sets:
            self.topology which is an OpenMM topology object
            self.crd which is the OpenMM coordinates object
        """

        if self.topology_format.upper() == "AMBER":
            if self.topology is None:
                self.top = app.AmberPrmtopFile(self.top_file)
                self.topology = self.top.topology

        elif self.topology_format.upper() == "GROMACS":
            if self.topology is None:
                if self.pbc == True:
                    assert self.angles[0] == self.angles[1] == self.angles[2] == 90.,\
                        "Only rectangular and cubic boxes are supported."
                    self.top = app.GromacsTopFile(self.top_file, periodicBoxVectors=(openmm.Vec3(self.cell[0], 0., 0.),
                                                                                openmm.Vec3(0., self.cell[1], 0.),
                                                                                openmm.Vec3(0, 0, self.cell[2])))
                else:
                    self.top = app.GromacsTopFile(self.top_file)
                self.topology = self.top.topology

        elif self.topology_format.upper() == "CHARMM":
            if self.topology is None:
                self.top = app.CharmmPsfFile(self.top_file)
                if self.pbc == True:
                    self.top.setBox(self.cell[0], self.cell[1], self.cell[2],
                                    alpha=self.angles[0]*unit.degree, beta=self.angles[1]*unit.degree,
                                    gamma=self.angles[2]*unit.degree)
                self.topology = self.top.topology

                p = Path(__file__).parent
                print(p)
                print(str(p)+'/top_all36_cgenff.rtf')
                self.top_params = app.CharmmParameterSet(str(p)+'/top_all36_cgenff.rtf', str(p)+'/par_all36_cgenff.prm',
                                                       str(p)+'/toppar_water_ions.str', self.charmm_param_file)

        elif self.topology_format == "OpenMM":
            if self.topology is None:
                self.top = app.ForceField(*self.xml_files)

        else:
            raise NotImplementedError("Topology format {} is not currently supported.".format(self.topology_format))

        # Read coordinate file
        if self.crd_format.upper() == "AMBER":
            crd = app.AmberInpcrdFile(self.crd_file)
        elif self.crd_format.upper() == "GROMACS":
            crd = app.GromacsGroFile(self.crd_file)
        elif self.crd_format.upper() == "CHARMM":
            crd = app.CharmmCrdFile(self.crd_file)
        elif self.crd_format.upper() == "PDB":
            crd = app.PDBFile(self.crd_file)
        else:
            raise NotImplementedError("Coordinate format {} is not currently supported.".format(self.crd_format))

        if crd != None:
            self.crd = crd

    def set_integrator(self):
        """
        sets the OpenMM MD integrator object

        Parameters
        ----------
        self.integrator_params
            "settings" to be used with the integrator
        """

        self.integrator = self.integrator(self.integrator_params['temperature'],
                                          self.integrator_params["frictionCoeff"],
                                          self.integrator_params["stepSize"])

    def set_platform(self):
        """
        sets the computational platform for OpenMM, GPU is preferred (muuuuch faster)
        """

        self.platform = openmm.Platform.getPlatformByName(self.platform_name)

    def create_openmm_system(self):
        """
        creates the OpenMM system object

        Parameters
        ----------
        self.topology_format :
            necessary bc OpenMM calls different functions to build the system based on the input formats
        """

        if self.topology_format.upper() == "CHARMM":
            assert self.top_params != None, 'please run import_molecular_system method first'
            self.system = self.top.createSystem(self.top_params, **self.create_system_params)

        elif self.topology_format == "OpenMM":
            molecule = self.crd
            self.system = self.top.createSystem(molecule.topology, **self.create_system_params)
            if self.pbc == True:
                self.system.setDefaultPeriodicBoxVectors(openmm.Vec3(self.cell[0], 0., 0.),
                                                         openmm.Vec3(0., self.cell[1], 0.),
                                                         openmm.Vec3(0, 0, self.cell[2]))


        elif (self.topology_format.upper() == "AMBER" or self.topology_format.upper() == "GROMACS") == True:
            self.system = self.top.createSystem(**self.create_system_params)


    def set_openmm_context(self): 
        """
        sets the OpenMM context object by checking constraints, reading the initial coordinates and bundling it all
        up w/ the integrator and platform (a.k.a. prepares the mdrun/classical calculation)

        Parameters
        ---------
        self.system : openmm.openmm.System object
        self.integrator : openmm.openmm.Integrator object
        self.platform : openmm.openmm.Platform object
        self.platform_properties : openmm.openmm.Platform settings of type dict
        self.crd : OpenMM coordinates object
        """

        print('Number of constraints: '+str(self.system.getNumConstraints())+'\nconstraints settings: '
              +str(self.create_system_params['constraints'])+'\nmake sure these are correct before setting context')

        """
        CHARMM PSF files have 'bonds' btwn H atoms of water. OpenMM deletes them, but applies constraints all the same
        if constraints=HBonds is set. These false constraints (every 3rd one) have to be removed before setting the 
        context or it will fail. 
        """

        if self.create_system_params['constraints'] == None:
            assert self.system.getNumConstraints() == 0, 'something wrong w/ the constraints'

            self.context = openmm.Context(self.system, self.integrator, self.platform, self.platform_properties)
            self.context.setPositions(self.crd.positions)

        if self.create_system_params['constraints'] == app.HBonds:
            """
            for n in range(0,30,1):
                print(str(self.system.getConstraintParameters(n)))
            for n in range(0,30,1):
                print(str(self.topology._bonds[n]))
            yes_choices = ['yes', 'y']
            no_choices = ['no', 'n']
            while True:
                constrain_correct = str(input('Are these constraints correct? (yes/no)\n'))
                if constrain_correct.lower() in yes_choices:
                    self.context = openmm.Context(self.system, self.integrator, self.platform, self.platform_properties)
                    self.context.setPositions(self.crd.positions)
                    break
                elif constrain_correct.lower() in no_choices:
                    print('please fix constraints using omm_sys.system.removeConstraint(index)')
                    break
                else:
                    print('Type yes or no')
                    continue
            """
            self.context = openmm.Context(self.system, self.integrator, self.platform, self.platform_properties)
            self.context.setPositions(self.crd.positions)

    def run_calculation(self, positions):
        """
        calculates classical energies and forces

        Parameters
        ----------
        positions : OpenMM coordinates object (OMM <Vec3> format) or numpy array, in nanometers

        self.context : OpenMM system context object

        returns:
            potential energies and forces (both numpy arrays)
        """

        assert self.context is not None, "please run OpenMM_system.set_opnemm_context first."

        self.context.setPositions(positions)

        """

        print('#############################################################')
        print('#       Calculating classical Energies & Forces             #')
        print('#############################################################')

        """

        epot = self.context.getState(getEnergy=True).getPotentialEnergy()._value #kJ/mol
        forces = self.context.getState(getForces=True).getForces(asNumpy=True)._value # in kJ/mol/nm

        """

        print('#############################################################')
        print('#                Calculation successful                     #')
        print('#############################################################')

        """

        return epot, forces
    
    def _extract_CustomNonbondedForce(self):
        """
        Automatically triggered if sigma == 1 and epsilon == 0 in NonbondedForce.
        In case some atom type used in the system has a NBFIX specified, OpenMM reads, stores, and handles the 
        nonbonded parameters differently. This function imports them and formats the in line with the traditional nonbonded parameters
        so that they reach the optimizer in the correct format. Code partially stolen from OpenMM's charmmpsffile.py
        """
        lj_index_list = [0 for atom in self.top.atom_list]
        num_lj_types = 0
        lj_r_sigma, lj_epsilon = [], [] 
        self.lj_type_list, atomtypes_list = [], []
        """
        looks like:
        vars(lj_type_list[0]) = {'name': 'CG331',
                                    'number': -1,
                                    'mass': Quantity(value=12.011, unit=dalton),
                                    'atomic_number': 6,
                                    'epsilon': -0.078,
                                    'rmin': 2.05,
                                    'epsilon_14': -0.01,
                                    'rmin_14': 1.9,
                                    'nbfix': {},
                                    'nbthole': {}}
        """
        for i, atom in enumerate(self.top.atom_list):

            atomtype = atom.type 

            if lj_index_list[i]:
                continue

            num_lj_types +=1
            lj_index_list[i] = num_lj_types
            ljtype = (atomtype.rmin, atomtype.epsilon)

            self.lj_type_list.append(atomtype) # collects dicts of atomtypes and params #TODO: filter out duplicates!
            atomtypes_list.append(atomtype.name) # collects strs of atomtypes
            lj_r_sigma.append(float(atomtype.rmin)) # rmin is r/sigma in Angström
            lj_epsilon.append(float(atomtype.epsilon)) # epsilon in kcal/mol

            for j in range(i+1, len(self.top.atom_list)):

                atomtype2 = self.top.atom_list[j].type

                if lj_index_list[j] > 0: 
                    continue # already assigned

                if atomtype2 is atomtype:
                    lj_index_list[j] = num_lj_types

                elif not atomtype.nbfix and not atomtype.nbthole and not atomtype2.nbfix and not atomtype2.nbthole:
                    # Only non-NBFIXed and non-NBTholed atom types can be compressed
                    ljtype2 = (atomtype2.rmin, atomtype2.epsilon)

                    if ljtype == ljtype2:
                        lj_index_list[j] = num_lj_types

        # Conversion factors from CHARMM to OpenMM/Gromacs/SI units
        length_conv = unit.angstrom.conversion_factor_to(unit.nanometer)
        ene_conv = unit.kilocalorie_per_mole.conversion_factor_to(unit.kilojoule_per_mole)

        # construct instances for atomtype-based custom nb params
        self.custom_nb_params = np.vstack((lj_r_sigma, lj_epsilon, atomtypes_list), dtype='object').T
        self.custom_nb_params[:,0] = self.custom_nb_params[:,0].astype('float') * length_conv #CAREFUL only converts unit -> compatibility w/ acoef&bcoef
        self.custom_nb_params[:,1] = self.custom_nb_params[:,1].astype('float') * ene_conv #CAREFUL only converts unit -> compatibility w/ acoef&bcoef

        nbfix_rij, nbfix_wdij = [], []
        nbfix_atype1, nbfix_atype2 = [], []

        for i in range(len(self.lj_type_list)):

            for j in range(len(self.lj_type_list)):

                atomtype_j = self.lj_type_list[j].name

                try:

                    #has NBFIX
                    rij, wdij = self.lj_type_list[i].nbfix[atomtype_j][:2]
                    atype1 = self.lj_type_list[i].name
                    atype2 = atomtype_j

                    nbfix_rij.append(rij)
                    nbfix_wdij.append(wdij)
                    nbfix_atype1.append(atype1)
                    nbfix_atype2.append(atype2)

                except KeyError:

                    # no NBFIX
                    continue

        self.nbfix = np.vstack((nbfix_rij, nbfix_wdij, nbfix_atype1, nbfix_atype2), dtype='object').T
        self.nbfix[:,0] = self.nbfix[:,0].astype('float') * length_conv
        self.nbfix[:,1] = self.nbfix[:,1].astype('float') * ene_conv


    def extract_forcefield(self):
        """
        Extracts all the force field params from the OpenMM system. 
        So far, HarmonicBondforce, HarmonicAngleForce, PeriodicTotrsionForce, and NonbondedForce are defined. NBExceptions
        do not work at the moment as they would require a Context update (terribly slow) in the parametrization loop
        (see https://github.com/openmm/openmm/issues/252).
        WARNING: Static implementation for only one CustomNonbondedForce in the OpenMM system. If there are more, this will not work.


        Parameters
        ----------
        self.system : OpenMM system object

        sets:
            self.force_group_idx_omm : list
                stores openmm force group number
            self.forces_indices : dict
                force group dict that lists the omm force groups and indices
        """

        assert self.system is not None, 'OpenMM system not set.'

        # stores openmm force group number
        self.force_group_idx_omm = []
        # force group dict that lists the omm force groups and indices
        self.forces_indices = {}

        force_idx = 0
        forces = self.system.getForces()

        for force in forces:

            # Get force group name
            force_key = force.__class__.__name__
            # Get force group number
            self.force_group_idx_omm.append(force.getForceGroup())

            if force_key not in self.forces_indices:
                self.forces_indices[force_key] = []

            self.forces_indices[force_key].append(force_idx)
            force_idx += 1

        # dict w/ force group names and their omm indices
        self.force_groups = copy.deepcopy(self.forces_indices)

        # grab all forces of all types from the openmm sys and file them in our extracted_ff
        for group in self.force_groups:

            force_key = group

            if force_key not in list(self.extracted_ff.keys()):
                self.extracted_ff[force_key] = []

            # loop needed in case of tuple index for force group
            for force_indices in self.force_groups[force_key]:
                ff_params = self.system.getForce(force_indices)

                if force_key == 'HarmonicBondForce':

                    # create structured array to store ff bond term params
                    sub_force_field = np.recarray((ff_params.getNumBonds()),
                                                  formats=['int', 'int', 'float', 'float'],
                                                    names=['atom1', 'atom2', 'bond_length', 'force_constant'])

                    # do for every bond
                    for bond_index in range(ff_params.getNumBonds()):

                        # extract atom indices, bond length /nm, and force constant /kJ/mol/nm^2
                        # & put 'em into the storage array
                        for i, value in enumerate(ff_params.getBondParameters(bond_index)):
                            # omm returns value directly
                            if i in [0, 1]:
                                sub_force_field[bond_index][i] = value
                            # omm returns value via _value method
                            elif i in [2, 3]:
                                sub_force_field[bond_index][i] = value._value

                    # Append sub_force_field to extracted_ff[force_key]
                    self.extracted_ff[force_key].append(sub_force_field)

                if force_key == 'HarmonicAngleForce':

                    # create structured array to store ff angle term params
                    sub_force_field = np.recarray((ff_params.getNumAngles()),
                                                  formats=['int', 'int', 'int', 'float', 'float'],
                                                    names=['atom1', 'atom2', 'atom3', 'angle', 'force_constant'])

                    # do for every angle
                    for angle_index in range(ff_params.getNumAngles()):

                        # extract atom indices, angle /rad, and force constant /kJ/mol/rad^2
                        # & put 'em into the storage array
                        for i, value in enumerate(ff_params.getAngleParameters(angle_index)):
                            # omm returns value directly
                            if i in [0, 1, 2]:
                                sub_force_field[angle_index][i] = value
                            # omm returns value via _value method
                            elif i in [3, 4]:
                                sub_force_field[angle_index][i] = value._value

                    # Append sub_force_field to extracted_ff[force_key]
                    self.extracted_ff[force_key].append(sub_force_field)

                if force_key == 'PeriodicTorsionForce':

                    # create structured array to store ff torsion term params
                    sub_force_field = np.recarray((ff_params.getNumTorsions()),
                                                  formats=['int', 'int', 'int', 'int', 'int', 'float', 'float'],
                                                    names=['atom1', 'atom2', 'atom3', 'atom4', 'periodicity', 'phase',
                                                             'force_constant'])

                    # do for every torsion
                    for torsion_index in range(ff_params.getNumTorsions()):

                        # extract atom indices, periodicity, phase /rad, and force constant /kJ/mol/rad^2
                        # & put 'em into the storage array
                        for i, value in enumerate(ff_params.getTorsionParameters(torsion_index)):
                            # omm returns value directly
                            if i in [0, 1, 2, 3, 4]:
                                sub_force_field[torsion_index][i] = value
                            # omm returns value via _value method
                            elif i in [5, 6]:
                                sub_force_field[torsion_index][i] = value._value

                    # Append sub_force_field to extracted_ff[force_key]
                    self.extracted_ff[force_key].append(sub_force_field)

                if force_key == 'NonbondedForce':

                    # create structured array to store ff nonbonded term params
                    sub_force_field = np.recarray((ff_params.getNumParticles()),
                                                  formats=['float', 'float', 'float'],
                                                    names=['charge', 'lj_sigma', 'lj_eps'])

                    # do for every particle
                    for particle_index in range(ff_params.getNumParticles()):

                        # extract charge, LJ sigma /nm, and LJ epsilon /kJ/mol
                        # & put 'em into the storage array
                        for i, value in enumerate(ff_params.getParticleParameters(particle_index)):

                            # omm returns value via _value method
                            sub_force_field[particle_index][i] = value._value

                    # Append sub_force_field to extracted_ff[force_key]
                    self.extracted_ff[force_key].append(sub_force_field)

                    #Check for NBFIX
                    if (np.unique(sub_force_field['lj_sigma']) == 1 or np.unique(sub_force_field['lj_eps']) == 0) == True:

                        customnb = self.system.getForce(self.force_groups['CustomNonbondedForce'][0])
                        acoeffunc = customnb.getTabulatedFunction(0)
                        bcoeffunc = customnb.getTabulatedFunction(1)
                        num_coeffs = len(acoeffunc.getFunctionParameters()[-1])

                        sub_force_field2 = np.recarray((num_coeffs),
                                                       formats=['float', 'float'],
                                                       names = ['acoef', 'bcoef'])
                        sub_force_field2['acoef'] = acoeffunc.getFunctionParameters()[-1] #skip xsize&ysize, only grab values
                        sub_force_field2['bcoef'] = bcoeffunc.getFunctionParameters()[-1]

                        if 'CustomNonbondedForce' not in list(self.extracted_ff.keys()):
                            self.extracted_ff['CustomNonbondedForce'] = []

                        self.extracted_ff['CustomNonbondedForce'].append(sub_force_field2)

                        self._extract_CustomNonbondedForce()
                        

                if force_key == 'NBException':

                    # create structured array to store ff special interaction term params
                    sub_force_field = np.recarray((ff_params.getNumExceptions()),
                                                  formats=['int', 'int', 'float', 'float', 'float'],
                                                  names=['atom1', 'atom2', 'chargeProd', 'sigma',
                                                         'epsilon'])
                    # do for every special interaction
                    for interaction_index in range(ff_params.getNumExceptions()):

                        # extract atom indices, charge /elementary charge**2, LJ sigma /nm, and LJ epsilon /kJ/mol
                        # & put 'em into the storage array
                        for i, value in enumerate(ff_params.getExceptionParameters(interaction_index)):
                            # omm returns value directly
                            if i in [0, 1]:
                                sub_force_field[interaction_index][i] = value
                            # omm returns value via _value method to strip off unit
                            elif i in [2, 3, 4]:
                                sub_force_field[interaction_index][i] = value._value

                    # Append sub_force_field to extracted_ff[force_key]
                    self.extracted_ff[force_key].append(sub_force_field)

    def write_extracted_ff(self, output_path=None):
        """
        writes the extracted force field parameters from the OpenMM system to file

        Parameters
        ----------
        output_path : str, optional, default=None
            path where the .ff files containing the force field parameters are written
        """
        if output_path != None:

            assert Path(output_path).is_dir() is True, 'output directory does not exist'

            if output_path[-1] != '/':
                output_path += '/'

        else:
            output_path = os.getcwd()+'/'

        for force_type in self.extracted_ff.keys():

            if force_type == 'HarmonicBondForce':

                column_fmt = ['%i', '%i', '%.5f', '%.1f']

            elif force_type == 'HarmonicAngleForce':

                column_fmt = ['%i', '%i', '%i', '%.8f', '%.4f']

            elif force_type == 'PeriodicTorsionForce':

                column_fmt = ['%i', '%i', '%i', '%i', '%i', '%.8f', '%.4f']

            elif force_type == 'NonbondedForce':

                column_fmt = ['%.3f', '%.8f', '%.8f']

            elif force_type == 'NBException':

                column_fmt = ['%i', '%i', '%.6f', '%.8f', '%.8f']

            if len(self.extracted_ff[force_type]) > 1:

                for i, array in enumerate(self.extracted_ff[force_type]):

                    fname = output_path + str(force_type) + '_' + str(i) + '.ff'

                    header_fmtd = str(force_type) + ' ' + str(self.force_groups[force_type][i]) + '\n'

                    for n, name in enumerate(array.dtype.names):
                        if n < len(array.dtype.names) - 1:
                            header_fmtd += str(name) + ' '
                        elif n == len(array.dtype.names) - 1:
                            header_fmtd += str(name)

                    np.savetxt(fname, array, delimiter=' ', newline='\n', fmt=column_fmt, header=header_fmtd)

            elif len(self.extracted_ff[force_type]) == 1:

                array = self.extracted_ff[force_type][0]

                fname = output_path + str(force_type) + '.ff'

                header_fmtd = str(force_type) + ' ' + str(self.force_groups[force_type][0]) + '\n'

                for n, name in enumerate(array.dtype.names):
                    if n < len(array.dtype.names) - 1:
                        header_fmtd += str(name) + ' '
                    elif n == len(array.dtype.names) - 1:
                        header_fmtd += str(name)

                np.savetxt(fname, array, delimiter=' ', newline='\n', fmt=column_fmt, header=header_fmtd)

        filenames = [x for x in fnmatch.filter(os.listdir(output_path), "*.ff") if '*opt*' not in x ]
        filenames.sort()

        assert 'extracted.ff' not in filenames, 'extracted.ff already exists'

        for filename in filenames:

            os.system('cat '+str(filename)+' >> extracted.ff')

        os.system('echo \# END >> extracted.ff')


    def write_ff_optimizable(self, output_path=None, file_name=None): 

        """
        writes the optimized force field parameters from the OpenMM system to file

        Parameters
        ----------
        output_path : str, optional, default=None
            path where the .ff files containing the force field parameters are written 
        file_name : str, optional, default=None
            filename of ...._opt.ff       
        """

        if output_path != None:

            assert Path(output_path).is_dir() is True, 'output directory does not exist'

            if output_path[-1] != '/':
                output_path += '/'
        
        else:
            output_path = os.getcwd()+'/'

        if file_name == None:

            file_name = 'ff_optimizable.ff'

        for force_type in self.ff_optimizable.keys():

            if force_type == 'HarmonicBondForce':

                column_fmt = ['%i', '%i', '%.5f', '%.1f']

            elif force_type == 'HarmonicAngleForce':

                column_fmt = ['%i', '%i', '%i', '%.8f', '%.4f']

            elif force_type == 'PeriodicTorsionForce':

                column_fmt = ['%i', '%i', '%i', '%i', '%i', '%.8f', '%.4f']

            elif force_type == 'NonbondedForce':

                column_fmt = ['%.3f', '%.8f', '%.8f']

            elif force_type == 'NBException':

                column_fmt = ['%i', '%i', '%.6f', '%.8f', '%.8f']

            if len(self.ff_optimizable[force_type]) > 1:

                for i, array in enumerate(self.extracted_ff[force_type]):

                    fname = output_path + file_name +'_'+ str(force_type) + '_' + str(i) + '_opt.ff'

                    header_fmtd = str(force_type) + ' ' + str(self.force_groups[force_type][i]) + '\n'

                    for n, name in enumerate(array.dtype.names):
                        if n < len(array.dtype.names) - 1:
                            header_fmtd += str(name) + ' '
                        elif n == len(array.dtype.names) - 1:
                            header_fmtd += str(name)

                    np.savetxt(fname, array, delimiter=' ', newline='\n', fmt=column_fmt, header=header_fmtd)

            elif len(self.ff_optimizable[force_type]) == 1:

                array = self.ff_optimizable[force_type][0]

                fname = output_path + file_name +'_'+ str(force_type) + '_opt.ff'

                header_fmtd = str(force_type) + ' ' + str(self.force_groups[force_type][0]) + '\n'

                for n, name in enumerate(array.dtype.names):
                    if n < len(array.dtype.names) - 1:
                        header_fmtd += str(name) + ' '
                    elif n == len(array.dtype.names) - 1:
                        header_fmtd += str(name)

                np.savetxt(fname, array, delimiter=' ', newline='\n', fmt=column_fmt, header=header_fmtd)

        filenames = fnmatch.filter(os.listdir(output_path), '*_opt.ff')
        filenames.sort()

        assert file_name not in filenames, '{} already exists'.format(file_name)

        for filename in filenames:

            os.system('cat '+str(filename)+' >> '+str(file_name)+'_ff_optimizable.ff')

        os.system('echo \# END >> '+str(file_name)+'_ff_optimizable.ff')


    def read_extracted_ff(self, input_path=None):
        """
        Read in an already existing/previously extracted extracted.ff file

        Parameters
        ----------
        input_path : str, optional, default=None
            path where the extracted.ff file containing previously extracted force field parameters is stored
        """

        if input_path != None:

            assert Path(input_path).is_dir() is True, 'input directory does not exist'

            if input_path[-1] != '/':
                input_path += '/'

        else:
            input_path = os.getcwd()+'/'

        extracted_ff_file = open(input_path+'extracted.ff', 'r')

        _extracted_ff = []

        for line_nr, line in enumerate(extracted_ff_file.readlines()):

            line = line.strip()
            _extracted_ff.append(line)

        extracted_ff_file.close()

        comment_line_nrs = []
        comment_lines = []
        for line_nr, line in enumerate(_extracted_ff):
            if '#' in line:
                comment_line_nrs.append(line_nr)
                comment_lines.append(line)

        self.read_in_ff = {}
        for force_type in self.force_groups_dict:
            for line in comment_lines:
                if force_type in line:
                    self.read_in_ff[force_type] = []

        self.force_groups_read = copy.deepcopy(self.read_in_ff)

        for group in self.force_groups_read:
            for line in comment_lines:
                if group in line:
                    self.force_groups_read[group].append(int(line[-1]))

        for group in self.force_groups_read:
            force_group_occurrence = np.argwhere((np.char.find(comment_lines, group)) != -1)

            for occurrence in force_group_occurrence:
                header = comment_line_nrs[int(occurrence)]+2
                footer = len(_extracted_ff)-header-(len(_extracted_ff)-comment_line_nrs[int(occurrence)+2])

                if group == 'HarmonicBondForce':
                    array = np.loadtxt(input_path+'extracted.ff', skiprows=header, max_rows=footer, dtype={'names':
                                                            ('atom1', 'atom2', 'bond_length', 'force_constant'),
                                                                                                'formats':
                                                            ('int', 'int', 'float', 'float')})

                elif group == 'HarmonicAngleForce':
                    array = np.loadtxt(input_path+'extracted.ff', skiprows=header, max_rows=footer, dtype={'names':
                                            ('atom1', 'atom2', 'atom3', 'angle', 'force_constant'),
                                                                                                'formats':
                                            ('int', 'int', 'int', 'float', 'float')})

                elif group == 'PeriodicTorsionForce':
                    array = np.loadtxt(input_path+'extracted.ff', skiprows=header, max_rows=footer, dtype={'names':
                                            ('atom1', 'atom2', 'atom3', 'atom4', 'periodicity', 'phase',
                                            'force_constant'),
                                                                                                'formats':
                                            ('int', 'int', 'int', 'int', 'int', 'float', 'float')})

                elif group == 'NBException':
                    array = np.loadtxt(input_path+'extracted.ff', skiprows=header, max_rows=footer, dtype={'names':
                                            ('atom1', 'atom2', 'chargeProd', 'sigma', 'epsilon'),
                                                                                                'formats':
                                            ('int', 'int', 'float', 'float', 'float')})

                elif group == 'NonbondedForce':
                    array = np.loadtxt(input_path+'extracted.ff', skiprows=header, max_rows=footer, dtype={'names':
                                                                        ('charge', 'lj_sigma', 'lj_eps'),
                                                                                                'formats':
                                                                        ('float', 'float', 'float')})
                                                                        # GMX/OMM units: nm and kJ/mol
                self.read_in_ff[group].append(array)

    def get_charges(self):
        """
        extracts the classical charges from the OpenMM system

        Parameters
        ----------
        self.extracted_ff

        sets:
            self.charges
        """

        if len(self.extracted_ff['NonbondedForce']) != 0:

            self.charges = self.extracted_ff['NonbondedForce'][0]['charge']

        elif len(self.extracted_ff['NonbondedForce']) == 0:
            assert len(self.read_in_ff['NonbondedForce']) != 0, 'No force field present. ' \
                                                                'Use method extract_forcefield or method ' \
                                                                'read_extracted_ff first.'

            self.charges = self.read_in_ff['NonbondedForce'][0]['charge']


    def write_ommsys2xml(self, output_path='.', fname='omm_system'):
    
        """

        Method that writes the OpenMM system stored in self.system to an XML file.

        Parameters
        ----------
        output_path : str, optional, default='.'
        path where the .xml file containing the system is written
        fname : str, optional, default='omm_system'
        filename of the .xml file

        Returns
        -------
        `True` if file was closed successfully. `False` otherwise.

        """

        from simtk.openmm import XmlSerializer

        if output_path != '.':

            assert Path(output_path).is_dir() is True, 'output directory does not exist'

            if output_path[-1] != '/':
                output_path += '/'

        else:
            output_path = os.getcwd()+'/'

        outfile = output_path + str(fname) + '.xml'

        serialized_system = XmlSerializer.serializeSystem(self.system)
        file = open(outfile, 'w')
        file.write(serialized_system)
        file.close()

        return file.close()

    def create_force_field_optimizable(self, opt_bonds=False, opt_angles=False, opt_torsions=False, opt_charges=False,
                                       opt_lj=False):

        """
        creates a writable dictionary to store the parameters that are to be optimized

        Parameters
        ----------
        opt_bonds : bool
            Flag that signals whether the bond parameters will be optimized.
        opt_angles : bool
             Flag that signals whether the angle parameters will be optimized.
        opt_torsions : bool
            Flag that signals whether the dihedral parameters will be optimized.
        opt_charges : bool
            Flag that signals whether the charges will be optimized.
        opt_lj : bool
            Flag that signals whether the Lennard-Jones parameter will be optimized.

        sets: self.ff_optimizable
        """

        assert self.extracted_ff is not None, "\t * ff_extracted dictionary was not created yet." \
                                              "Run OpenMM_system.extract_forcefield method before."

        self.ff_optimizable = {}

        for force_type in self.extracted_ff.keys():

            # check if force type is optimizable and copy if so
            if (force_type == 'HarmonicBondForce' and opt_bonds == True) == True:

                self.ff_optimizable[force_type] = copy.deepcopy(self.extracted_ff[force_type])

            elif (force_type == 'HarmonicAngleForce' and opt_angles ==True) == True:

                self.ff_optimizable[force_type] = copy.deepcopy(self.extracted_ff[force_type])

            elif (force_type == 'PeriodicTorsionForce' and opt_torsions) == True:

                self.ff_optimizable[force_type] = copy.deepcopy(self.extracted_ff[force_type])

            elif force_type == 'NonbondedForce':

                if (opt_charges == True and opt_lj == True) == True:

                    if 'CustomNonbondedForce' in list(self.extracted_ff.keys()): #NBFIX check

                        assert self.extracted_ff['CustomNonbondedForce'] != [], 'NBFIX may be present but has not been extracted, check ff parameters'
                        
                        self.ff_optimizable['CustomNonbondedForce'] = copy.deepcopy(self.extracted_ff['CustomNonbondedForce'])
                    
                    self.ff_optimizable[force_type] = copy.deepcopy(self.extracted_ff[force_type])

                elif (opt_charges == True and opt_lj == False) == True:

                    optimizable_charges = copy.deepcopy(self.extracted_ff[force_type][0].charge)

                    nonbond_array = np.core.rec.fromarrays([optimizable_charges],
                                    shape=len(optimizable_charges),
                                                dtype=np.dtype(('charge', np.float)),
                                                names=['charge'])

                    self.ff_optimizable[force_type] = []
                    self.ff_optimizable[force_type].append(copy.deepcopy(nonbond_array))

                elif (opt_charges == False and opt_lj == True) == True:

                    optimizable_sigma = copy.deepcopy(self.extracted_ff[force_type][0].lj_sigma)
                    optimizable_eps = copy.deepcopy(self.extracted_ff[force_type][0].lj_eps)

                    #optimizable_charges = copy.deepcopy(self.extracted_ff[force_type][0].charge)
                    nonbond_array = np.rec.fromarrays([optimizable_sigma, optimizable_eps],
                                                           shape=len(optimizable_sigma),
                                                dtype=np.dtype([('lj_sigma', np.float64),('lj_eps', np.float64)]),
                                                names=['lj_sigma', 'lj_eps'])

                    self.ff_optimizable[force_type] = []
                    self.ff_optimizable[force_type].append(copy.deepcopy(nonbond_array))

            elif force_type == 'NBException':

                if (opt_charges == True and opt_lj == True) == True:
                    self.ff_optimizable[force_type] = copy.deepcopy(self.extracted_ff[force_type])

                elif (opt_charges == True and opt_lj == False) == True:
                    optimizable_chargeProd = copy.deepcopy(self.extracted_ff[force_type][0].chargeProd)
                    nb_exc_array = np.core.rec.fromarrays([optimizable_chargeProd],
                                                           shape=len(optimizable_chargeProd),
                                                           dtype=np.dtype(('chargeProd', np.float)),
                                                           names=['chargeProd'])

                    self.ff_optimizable[force_type] = []
                    self.ff_optimizable[force_type].append(copy.deepcopy(nb_exc_array))

                elif (opt_charges == False and opt_lj == True) == True:

                    optimizable_exc_sigma = copy.deepcopy(self.extracted_ff[force_type][0].sigma)
                    optimizable_exc_eps = copy.deepcopy(self.extracted_ff[force_type][0].epsilon)

                    nb_exc_array = np.rec.fromarrays([optimizable_exc_sigma, optimizable_exc_eps],
                                                           shape=len(optimizable_exc_sigma),
                                                dtype=np.dtype([('sigma', np.float64),('epsilon', np.float64)]),
                                                names=['sigma', 'epsilon'])

                    self.ff_optimizable[force_type] = []
                    self.ff_optimizable[force_type].append(copy.deepcopy(nb_exc_array))


    def set_parameters(self):
        """
        writes the optimized parameters to the OpenMM system

        Parameters
        ----------
        self.ff_optimizable : dictionary containing the optimizable force field parameters
        self.system : OpenMM system object

        """

        assert len(self.ff_optimizable) != 0, 'ff_optimizable does not exist, please create it first.'

        for force_key in self.ff_optimizable.keys():

            parameter_array_list = self.ff_optimizable[force_key]

            # loop needed in case of tuple index for force group
            for index_no, omm_force_index in enumerate(self.force_groups[force_key]):
                force = self.system.getForce(omm_force_index) # this is the openmm force object

                for index, parameters in enumerate(parameter_array_list[index_no]):

                    if force_key == 'NonbondedForce':
                        force.setParticleParameters(index, parameters["charge"], parameters["lj_sigma"], parameters["lj_eps"])
                        force.updateParametersInContext(self.context)

                    elif force_key == 'HarmonicBondForce':
                        force.setBondParameters(index, parameters["atom1"], parameters["atom2"], parameters["bond_length"],
                                                parameters['force_constant'])
                        force.updateParametersInContext(self.context)

                    elif force_key == 'HarmonicAngleForce':
                        force.setAngleParameters(index, parameters["atom1"], parameters["atom2"], parameters['atom3'], parameters["angle"],
                                                parameters['force_constant'])
                        force.updateParametersInContext(self.context)

                    elif force_key == 'PeriodicTorsionForce':
                        force.setTorsionParameters(index, parameters["atom1"], parameters["atom2"], parameters['atom3'], parameters['atom4'],
                                                parameters["periodicity"], parameters['phase'], parameters['force_constant'])
                        force.updateParametersInContext(self.context)

                    elif force_key == 'NBException':

                        force.setExceptionParameters(index, parameters["atom1"], parameters["atom2"], parameters["chargeProd"], parameters['sigma'],
                                                    parameters['epsilon'])
                        force.updateParametersInContext(self.context)
                
                if force_key == 'CustomNonbondedForce':

                    acoeffunc = force.getTabulatedFunction(0)
                    bcoeffunc = force.getTabulatedFunction(1)

                    xsize, ysize = acoeffunc.getFunctionParameters()[:2]

                    acoeffunc.setFunctionParameters(xsize, ysize, parameter_array_list[index_no]['acoef']) #TODO: we got a dtype issue
                    bcoeffunc.setFunctionParameters(xsize, ysize, parameter_array_list[index_no]['bcoef'])

                    force.updateParametersInContext(self.context)
                




