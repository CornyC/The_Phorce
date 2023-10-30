import importlib
from System.input_paths import *
from OMM_interface.openmm import *
import os
import shutil
from VMD import VMD
import importlib
import os
import numpy as np
from pathlib import Path
import re
from ASE_interface.ase_calculation import CP2K
from Coord_Toolz.mdanalysis import MDA_reader
from Direct_cp2k_calculation.direct_cp2k import Direct_Calculator
from System.input_paths import Input_Paths
from System.system import Molecular_system
from ASE_interface.ase_calculation import ASE_system
from Parametrization.parametrization import Parametrization
from Parametrization.optimizers import Optimizer
from System.system import write_single_pdb
from OMM_interface.openmm import OpenMM_system

# set paths
input_paths = Input_Paths()
input_paths.mm_top_file = 'smallbox.psf'
input_paths.mm_top_file_path = '/home/ac127777/Documents/methylphosphate/cp2k_geom_opts/mp0mp0/frame0/'
input_paths.crd_pdb_file = 'smallbox.pdb'
input_paths.crd_pdb_file_path = input_paths.mm_top_file_path
input_paths.str_file = 'mp0.str'
input_paths.str_file_path = '../methylphosphate/force_matching/test/'
input_paths.traj_file = 'optcoords.xyz'
input_paths.traj_file_path = '/home/ac127777/Documents/methylphosphate/cp2k_geom_opts/mp0mp0/'
input_paths.working_dir = '../methylphosphate/force_matching/test/'
input_paths.project_name = 'mp0mp0'
input_paths.set()

# set up the openmm system
omm_sys = OpenMM_system(topology_format='CHARMM', top_file=input_paths.mm_top, crd_format='PDB',
                        crd_file=input_paths.crd_pdb, charmm_param_file=input_paths.stream, pbc=True,
                        cell=[1.6, 1.6, 1.6], angles=[90, 90, 90])

omm_sys.create_system_params['nonbondedCutoff'] = 0.8 * unit.nanometer

omm_sys.create_system_params['constraints'] = app.HBonds

omm_sys.import_molecular_system()

omm_sys.set_integrator()

omm_sys.set_platform()

omm_sys.create_openmm_system()

omm_sys.set_openmm_context()

omm_sys.extract_forcefield()

#omm_sys.read_extracted_ff(input_path='/home/ac127777/Documents/The_Force')

omm_sys.get_charges()
# set up ase sys
from Coord_Toolz.mdanalysis import *

mdr = MDA_reader()

mdr.set_traj_input(input_paths.crd_pdb, input_paths.traj)

trajcoords=get_coords(mdr.universe.atoms)

from ASE_interface.ase_calculation import *

# prepare net force calc
mdr.molecule1_solv = mdr.delete_one_molecule('not resid 2')
mdr.molecule2_solv = mdr.delete_one_molecule('not resid 1')
mdr.molecule_only = mdr.universe.select_atoms('resid 1 or resid 2')
coords1=get_coords(mdr.molecule1_solv)
coords2=get_coords(mdr.molecule2_solv)

inp = """
&FORCE_EVAL
  METHOD Quickstep              ! Electronic structure method (DFT,...)
  &DFT
    BASIS_SET_FILE_NAME  BASIS_MOLOPT_UZH
    POTENTIAL_FILE_NAME  POTENTIAL_UZH
    CHARGE 0
    &MGRID
      NGRIDS 5
      CUTOFF 550
      REL_CUTOFF 80
    &END MGRID
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &END QS
    &POISSON                    ! Solver requested for non periodic calculations
      PERIODIC XYZ
      PSOLVER  PERIODIC          ! Type of solver
    &END POISSON
    &SCF                        ! Parameters controlling the convergence of the scf. This section should not be changed. 
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-5
      MAX_SCF 800
      &MIXING
        ALPHA 0.4
      &END MIXING
      &OT
        MINIMIZER CG
      &END OT
    &END SCF
    &XC                        ! Parameters needed to compute the electronic exchange potential 
      &VDW_POTENTIAL
        DISPERSION_FUNCTIONAL PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3
          PARAMETER_FILE_NAME  dftd3.dat
          REFERENCE_FUNCTIONAL PBE
          CALCULATE_C9_TERM TRUE
          REFERENCE_C9_TERM TRUE
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
  &END DFT

  &SUBSYS
    &KIND H
      ELEMENT H
      BASIS_SET TZV2P-MOLOPT-PBE-GTH-q1
      POTENTIAL GTH-PBE-q1
    &END KIND
    &KIND C
      ELEMENT C
      BASIS_SET TZV2P-MOLOPT-PBE-GTH-q4
      POTENTIAL GTH-PBE-q4
    &END KIND
    &KIND P
      ELEMENT P
      BASIS_SET TZV2P-MOLOPT-PBE-GTH-q5
      POTENTIAL GTH-PBE-q5
    &END KIND
    &KIND O
      ELEMENT O
      BASIS_SET TZV2P-MOLOPT-PBE-GTH-q6
      POTENTIAL GTH-PBE-q6
    &END KIND
    &PRINT
      &ATOMIC_COORDINATES LOW
      &END ATOMIC_COORDINATES
      &INTERATOMIC_DISTANCES LOW
      &END INTERATOMIC_DISTANCES
      &KINDS 
        POTENTIAL
      &END KINDS
      &TOPOLOGY_INFO
        PSF_INFO
      &END TOPOLOGY_INFO
    &END PRINT
  &END SUBSYS

  &PRINT
    &FORCES ON
    &END FORCES
  &END PRINT

&END FORCE_EVAL """

ase_sys = ASE_system(mdr.molecule2_solv.elements, coords1[0])
ase_sys.cell = ([16.0, 16.0, 16.0])
ase_sys.pbc = ([True, True, True])
ase_sys.construct_atoms_object()


ase_sys = ASE_system(mdr.universe.atoms.elements, trajcoords[0])

ase_sys.cell = ([16.0, 16.0, 16.0])

ase_sys.pbc = ([True, True, True])

ase_sys.construct_atoms_object()

"""
for frame_nr, frame in enumerate(coords1):
    ase_sys = ASE_system(mdr.molecule1_solv.elements, frame)
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
                inp=inp,
                print_level='MEDIUM')
    CP2K.command = 'env OMP_NUM_THREADS=6 cp2k_shell.ssmp'
    ase_sys.atoms.calc = calc
    ase_sys.run_calculation(run_type='single_point')
    np.savetxt('forces_energy_mol1_frame'+str(frame_nr)+'.txt', ase_sys.forces, header='E: '+str(ase_sys.energy))
    outstr='cp cp2k.out mol1_frame'+str(frame_nr)+'.out'
    os.system(outstr)
    os.system('rm cp2k.out')
    os.system('pkill cp2k_shell.ssmp')
    os.system('touch cp2k.out')

for frame_nr, frame in enumerate(coords2):
    ase_sys = ASE_system(mdr.molecule2_solv.elements, frame)
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
                inp=inp,
                print_level='MEDIUM')
    CP2K.command = "env OMP_NUM_THREADS=6 cp2k_shell.ssmp"
    ase_sys.atoms.calc = calc
    ase_sys.run_calculation(run_type='single_point')
    np.savetxt('forces_energy_mol2_frame'+str(frame_nr)+'.txt', ase_sys.forces, header='E: '+str(ase_sys.energy))
    outstr='cp cp2k.out mol2_frame'+str(frame_nr)+'.out'
    os.system(outstr)
    os.system('rm cp2k.out')
    os.system('pkill cp2k_shell.ssmp')
    os.system('touch cp2k.out')
"""
# grab dft data

from System.system import *
molsys = Molecular_system(mdr.universe, ase_sys, omm_sys)
# define molecules
molsys.molecule1 = mdr.molecule1_solv
molsys.molecule2 = mdr.molecule2_solv
molsys.naked_molecule = mdr.molecule_only
molsys.naked_molecule_n_atoms = molsys.naked_molecule.n_atoms

#TODO1: write general func to do this


# Define a function to copy a file
def perform_ase_calculation(input_paths, coords, frame_nr, molecule_solv, basis_set, inp):
    ase_sys = ASE_system(molecule_solv.elements, coords)
    ase_sys.cell = ([16.0, 16.0, 16.0])
    ase_sys.pbc = ([True, True, True])
    ase_sys.construct_atoms_object()

    calc = CP2K(
        basis_set=basis_set,
        inp=inp,
        print_level='MEDIUM'
    )

    CP2K.command = 'env OMP_NUM_THREADS=6 cp2k_shell.ssmp'
    ase_sys.atoms.calc = calc
    ase_sys.run_calculation(run_type='single_point')

    np.savetxt(f'forces_energy_mol{frame_nr}.txt', ase_sys.forces, header=f'E: {ase_sys.energy}')
    outstr = f'cp cp2k.out mol{frame_nr}.out'
    os.system(outstr)
    os.system('rm cp2k.out')
    os.system('pkill cp2k_shell.ssmp')
    os.system('touch cp2k.out')

    return ase_sys
# def copy_file(source_path, destination_path):
#    try:
#        shutil.copy(source_path, destination_path)
#        print(f"File copied from {source_path} to {destination_path}.")
#    except Exception as e:
#        print(f"Error copying file: {e}")

# Temp example:
# source_file = 'source.txt'
# destination_file = 'destination.txt'
# copy_file(source_file, destination_file)


# init arrays
molsys.qm_forces = np.zeros((len(trajcoords), trajcoords.shape[1], 3))
molsys.qm_energies = np.zeros((len(trajcoords), 1))

mol1_fqm = np.zeros((len(coords1), coords1.shape[1], 3))
mol2_fqm = np.zeros((len(coords2), coords2.shape[1], 3))
mol1_eqm = np.zeros((len(coords1), 1))
mol2_eqm = np.zeros((len(coords2), 1))
# fill arrays
for n, frane in enumerate(coords1):
    mol1_fqm[n, :, :] = np.genfromtxt(str(input_paths.traj_file_path)+'/frame'+str(n)+'/forces_energy_mol1_frame'+str(n)+'.txt', skip_header=1)
    f = open(str(input_paths.traj_file_path)+'/frame'+str(n)+'/forces_energy_mol1_frame'+str(n)+'.txt', 'r')
    read=[]
    for i, line in enumerate(f.readlines()):
        line = line.strip('# E:\n')
        if i == 0:
            read.append(line)
    f.close()
    mol1_eqm[n] = float(read[0])

for n, frane in enumerate(coords2):
    mol2_fqm[n, :, :] = np.genfromtxt(str(input_paths.traj_file_path)+'/frame'+str(n)+'/forces_energy_mol2_frame'+str(n)+'.txt', skip_header=1)
    f = open(str(input_paths.traj_file_path)+'/frame'+str(n)+'/forces_energy_mol2_frame'+str(n)+'.txt', 'r')
    read=[]
    for i, line in enumerate(f.readlines()):
        line = line.strip('# E:\n')
        if i == 0:
            read.append(line)
    f.close()
    mol2_eqm[n] = float(read[0])
# define atom groups
mol1_only = mdr.universe.select_atoms('resid 1')
mol2_only = mdr.universe.select_atoms('resid 2')
# init arrays
fqm_raw = np.zeros((len(trajcoords), trajcoords.shape[1], 3))
eqm_raw = np.zeros((len(trajcoords), 1))
fqm_net = np.zeros((len(trajcoords), trajcoords.shape[1], 3))
eqm_net = np.zeros((len(trajcoords), 1))

fmm_raw = np.zeros((len(trajcoords), trajcoords.shape[1], 3))
emm_raw = np.zeros((len(trajcoords), 1))
fmm_net = np.zeros((len(trajcoords), trajcoords.shape[1], 3))
emm_net = np.zeros((len(trajcoords), 1))

from pathlib import Path
import re
# fill arrays
for n, frame in enumerate(trajcoords):
    p = Path(str(input_paths.traj_file_path)+'frame'+str(n)+'/geom_opt_rst.out')
    if p.exists():
        filepath = str(input_paths.traj_file_path)+'frame'+str(n)+'/geom_opt_rst.out'
    else:
        filepath = str(input_paths.traj_file_path)+'frame'+str(n)+'/geom_opt.out'
    f = open(filepath, 'r')
    read = []
    for index, line in enumerate(f.readlines()):
        line = line.strip()
        read.append(line)
    f.close()

    forces_start = [index for index, string in enumerate(read) if 'ATOMIC FORCES in [a.u.]' in string]
    forces = np.loadtxt(filepath, skiprows=forces_start[0] + 3, max_rows=22, usecols=(3, 4, 5), dtype=float) \
             # * 49614.752589482  # a.u. to kJ/mol/nm

    fqm_raw[n, :22, :] = forces

    energy_line = [index for index, string in enumerate(read) if 'ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:'
                   in string]

    energy = float(re.findall(r"[-+]?(?:\d*\.*\d+)", read[energy_line[-1]])[0]) * 2625.4996394798  # Hartree to kJ/mol

    eqm_raw[n] = energy

mol_fqm = np.concatenate((mol1_fqm[:,:11,:],mol2_fqm[:,:11,:]), axis=1)
wat_fqm = mol1_fqm[:,11:,:]-mol2_fqm[:,11:,:]
molwat_fqm = np.concatenate((mol_fqm[:,:,:],wat_fqm[:,:,:]),axis=1)
fqm_net = (fqm_raw - molwat_fqm) #* 49614.752589482  a.u. to kJ/mol/nm

# grab mm data

mol1_fmm = np.zeros((len(coords1), coords1.shape[1], 3))
mol2_fmm = np.zeros((len(coords2), coords2.shape[1], 3))
mol1_emm = np.zeros((len(coords1), 1))
mol2_emm = np.zeros((len(coords2), 1))

#TODO2 generate a .psf for mol1_solv and mol2_solv -.-

# Tried using VMD but this is complex process to generate a .psf file as it involves multiple softwares
# Can also be done using CHARMM (trying to find a easy way for this)
# Path to VMD executable
vmd_path = '/path/to/vmd'

# Path to the molecule's PDB file
pdb_file = '/path/to/molecule.pdb'

# Path to the output .psf file
psf_file = '/path/to/output.psf'

# Create a VMD instance
vmd = VMD(vmd_path)

# Load the PDB file
vmd.load_molecule(pdb_file)

# Save the molecule as a PSF file
vmd.save_psf(psf_file)
vmd.exit()

# or
    # define a function to create a .psf file
    # def create_psf_file(molecule, output_file_path):
    #try:
        # trying to find a way to extract atom and connectivity information from the molecule
        # and then we write it to the .psf file format.
        # Example: molecule_to_psf(molecule, output_file_path)
    #    print(f".psf file created: {output_file_path}")
    #    except Exception as e:
    #    print(f"Error creating .psf file: {e}")

# Temp example:
#psf_output_path = 'mol1_solv.psf'
#new_varnew_var = create_psf_file(mol1_solv, psf_output_path)



#TODO3: write general function to do this

def update_file_with_data(file_path, data_to_append):
    try:
        with open(file_path, 'a') as file:
            file.write(data_to_append)
        print("Data appended to the file successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Temp example:
file_path = "/path/to/your/file.txt"  # need to replace with the actual file path
data_to_append = "Your data to append"
update_file_with_data(file_path, data_to_append)

#TODO4: write function to update str file w/ DFT data if/as soon as DFT data is present
# Define a function to update a file with DFT data
def update_str_with_dft_data(str_file_path, dft_data):
    try:
        with open(str_file_path, 'a') as str_file:
            str_file.write(dft_data)
        print("STR file updated with DFT data.")
    except Exception as e:
        print(f"Error updating STR file: {e}")

# Temp example:
str_file_path = 'input.str'
dft_data = 'DFT data...'
update_str_with_dft_data(str_file_path, dft_data)

for frame_nr, frame in enumerate(coords1):
    mdr.molecule1_solv.atoms.positions = frame
    write_single_pdb(mdr.molecule1_solv.atoms, pdb_filename='optpos_phos1.pdb', pdb_path=input_paths.traj_file_path+'frame'+str(frame_nr))

    oms_p1_h2o = OpenMM_system(topology_format='CHARMM',
    top_file = input_paths.traj_file_path + 'frame0/optpos_phos1.psf', crd_format = 'PDB',
    crd_file = input_paths.traj_file_path + 'frame' + str(frame_nr) + '/optpos_phos1.pdb',
    charmm_param_file = '../methylphosphate/force_matching/test/mp0.str',
    pbc = True, cell = [1.6, 1.6, 1.6], angles = [90, 90, 90])

    oms_p1_h2o.import_molecular_system()
    oms_p1_h2o.set_integrator()
    oms_p1_h2o.set_platform()

    oms_p1_h2o.create_system_params = {'nonbondedMethod': app.PME, 'nonbondedCutoff': 0.8 * unit.nanometer,
                                         'switchDistance': 0.7 * unit.nanometer, 'constraints': app.HBonds,
                                         'rigidWater': False, 'temperature': 310.0 * unit.kelvin}

    oms_p1_h2o.create_openmm_system()
    oms_p1_h2o.set_openmm_context()
    oms_p1_h2o.extract_forcefield()

    ff_opt = copy.deepcopy(oms_p1_h2o.extracted_ff)

    force = oms_p1_h2o.system.getForce(6)
    terms = ff_opt['NonbondedForce']

    for index, term in enumerate(terms[0]):
        force.setParticleParameters(index, term["charge"], term["lj_sigma"], term["lj_eps"])
        force.updateParametersInContext(oms_p1_h2o.context)

    oms_p1_h2o.run_calculation(oms_p1_h2o.crd.positions)

    mol1_emm[frame_nr] = oms_p1_h2o.energies

    mol1_fmm[frame_nr, :, :] = oms_p1_h2o.forces

for frame_nr, frame in enumerate(coords2):
    mdr.molecule2_solv.atoms.positions = frame
    write_single_pdb(mdr.molecule2_solv.atoms, pdb_filename='optpos_phos2.pdb', pdb_path=input_paths.traj_file_path+'frame'+str(frame_nr))

    oms_p2_h2o = OpenMM_system(topology_format='CHARMM',
    top_file = input_paths.traj_file_path + 'frame0/optpos_phos2.psf', crd_format = 'PDB',
    crd_file = input_paths.traj_file_path + 'frame' + str(frame_nr) + '/optpos_phos2.pdb',
    charmm_param_file = '../methylphosphate/force_matching/test/mp0.str',
    pbc = True, cell = [1.6, 1.6, 1.6], angles = [90, 90, 90])

    oms_p2_h2o.import_molecular_system()
    oms_p2_h2o.set_integrator()
    oms_p2_h2o.set_platform()

    oms_p2_h2o.create_system_params = {'nonbondedMethod': app.PME, 'nonbondedCutoff': 0.8 * unit.nanometer,
                                         'switchDistance': 0.7 * unit.nanometer, 'constraints': app.HBonds,
                                         'rigidWater': False, 'temperature': 310.0 * unit.kelvin}

    oms_p2_h2o.create_openmm_system()
    oms_p2_h2o.set_openmm_context()
    oms_p2_h2o.extract_forcefield()

    ff_opt = copy.deepcopy(oms_p2_h2o.extracted_ff)

    force = oms_p2_h2o.system.getForce(6)
    terms = ff_opt['NonbondedForce']

    for index, term in enumerate(terms[0]):
        force.setParticleParameters(index, term["charge"], term["lj_sigma"], term["lj_eps"])
        force.updateParametersInContext(oms_p2_h2o.context)

    oms_p2_h2o.run_calculation(oms_p2_h2o.crd.positions)

    mol2_emm[frame_nr] = oms_p2_h2o.energies

    mol2_fmm[frame_nr, :, :] = oms_p2_h2o.forces


for frame_nr, frame in enumerate(trajcoords):
    mdr.universe.atoms.positions = frame
    write_single_pdb(mdr.universe.atoms, pdb_filename='optpos.pdb', pdb_path=input_paths.traj_file_path+'frame'+str(frame_nr))

    omm_sys = OpenMM_system(topology_format='CHARMM',
    top_file = input_paths.traj_file_path + 'frame0/smallbox.psf', crd_format = 'PDB',
    crd_file = input_paths.traj_file_path + 'frame' + str(frame_nr) + '/optpos.pdb',
    charmm_param_file = '../methylphosphate/force_matching/test/mp0.str',
    pbc = True, cell = [1.6, 1.6, 1.6], angles = [90, 90, 90])

    omm_sys.import_molecular_system()
    omm_sys.set_integrator()
    omm_sys.set_platform()

    omm_sys.create_system_params = {'nonbondedMethod': app.PME, 'nonbondedCutoff': 0.8 * unit.nanometer,
                                         'switchDistance': 0.7 * unit.nanometer, 'constraints': app.HBonds,
                                         'rigidWater': False, 'temperature': 310.0 * unit.kelvin}

    omm_sys.create_openmm_system()
    omm_sys.set_openmm_context()
    omm_sys.extract_forcefield()

    ff_opt = copy.deepcopy(omm_sys.extracted_ff)

    force = omm_sys.system.getForce(6)
    terms = ff_opt['NonbondedForce']

    for index, term in enumerate(terms[0]):
        force.setParticleParameters(index, term["charge"], term["lj_sigma"], term["lj_eps"])
        force.updateParametersInContext(omm_sys.context)

    omm_sys.run_calculation(omm_sys.crd.positions)

    emm_raw[frame_nr] = omm_sys.energies

    fmm_raw[frame_nr, :, :] = omm_sys.forces


mol_fmm = np.concatenate((mol1_fmm[:,:11,:],mol2_fmm[:,:11,:]), axis=1)
wat_fmm = mol1_fmm[:,11:,:]-mol2_fmm[:,11:,:]
molwat_fmm = np.concatenate((mol_fmm[:,:,:],wat_fmm[:,:,:]),axis=1)
fmm_net = (fmm_raw - molwat_fmm) #* 49614.752589482  a.u. to kJ/mol/nm

omm_sys.get_charges()

# pass on to molsys
molsys.mm_forces=fmm_net
molsys.qm_forces=fqm_net
molsys.mm_energies=emm_raw
molsys.qm_energies=eqm_raw

# get qm charges
from Direct_cp2k_calculation.direct_cp2k import *
cp2k_dc = Direct_Calculator(input_paths)
cp2k_dc.project_path = input_paths.traj_file_path
# create array
qmcharges = np.zeros((len(trajcoords), trajcoords.shape[1]))

# fill array
for frame_nr, frame in enumerate(trajcoords):
    p = Path(str(input_paths.traj_file_path)+'frame'+str(frame_nr)+'/geom_opt_rst.out')
    if p.exists():
        filepath = str(input_paths.traj_file_path)+'frame'+str(frame_nr)+'/geom_opt_rst.out'
    else:
        filepath = str(input_paths.traj_file_path)+'frame'+str(frame_nr)+'/geom_opt.out'
    f = open(filepath, 'r')
    read = []
    for index, line in enumerate(f.readlines()):
        line = line.strip()
        read.append(line)
    f.close()

    resp_start = [index for index, string in enumerate(read) if 'RESP charges:' in string]
    charges = np.loadtxt(filepath,
                         skiprows=resp_start[0] + 3,
                         max_rows=22, usecols=3, dtype=float)
    qmcharges[frame_nr, :22] = charges[:22]

omm_sys.create_force_field_optimizable(opt_lj=True)

molsys.generate_weights()

from Parametrization.optimizers import *

opti = Optimizer('scipy',"method='BFGS'")

from Parametrization.parametrization import *

Parametrization=Parametrization(molsys, opti)

Parametrization.fmm=Parametrization.fmm[:,:22,]
Parametrization.fqm=Parametrization.fqm[:,:22,]
Parametrization.n_atoms=22
Parametrization.ff_optimizable = omm_sys.ff_optimizable
Parametrization.calc_scaling_constants()

"""

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
        inp=inp,
        print_level='MEDIUM')

ase_sys.atoms.calc = calc


# bundle all in system
from System.system import *

molsys=Molecular_system(trajcoords, ase_sys, omm_sys)
# extract relevant data
molsys.naked_molecule_n_atoms = 22

omm_sys.run_calculation(omm_sys.crd.positions)

omm_sys.write_ommsys2xml()

molsys.mm_forces=molsys.openmm_sys.forces

mdr.remove_water_ions()

molsys.naked_molecule = mdr.molecule_only

from Direct_cp2k_calculation.direct_cp2k import *

cp2k_calc = Direct_Calculator(project_path='/home/ac127777/Documents/methylphosphate/cp2k_geom_opts/mp0mp0/testframe/0_fixed_h2o_geom_opt', project_name='mp0_geom_opt', run_type='charges')

cp2k_calc.extract_charges(charge_type='RESP', n_atoms=molsys.naked_molecule.n_atoms)

molsys.mm_charges_gp = molsys.openmm_sys.charges[:molsys.naked_molecule_n_atoms]

"""

#TODO5 do the params go into the optimizers? Do they work?