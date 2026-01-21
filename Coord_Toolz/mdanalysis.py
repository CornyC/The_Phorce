#### package imports ####

import MDAnalysis as mda
import MDAnalysis.core.groups
from MDAnalysis.core.universe import *
from MDAnalysis.analysis.base import AnalysisFromFunction
from MDAnalysis.coordinates.memory import MemoryReader #faster processing for small trajectories
import os
from pathlib import Path
import numpy as np
from System.paths import *


class MDA_reader:
    """
    Reads in sampled structures trajectory and topology. Coordinates in Angström. Allows for manipulations of atoms.
    Input upon initialization can be empty or a predefined MDAnalysis Universe: 
    Example: MDR = MDA_reader(MDAnalysis.core.universe.Universe) or MDR = MDA_reader()

    Parameters
    ----------
    all : MDAnalysis.core.universe.Universe object
        Contains all atoms and their names & identifiers, positions, forces...(check MDAnalysis doc)
    nosol : MDAnalysis.core.groups.AtomGroup
        Contains atom selection of universe that represents the molecule to parametrize w/o solvent or ions. Enables 
        calculation of forces between molecule and solvent. 
    mol1 : MDAnalysis.core.groups.AtomGroup
        Contains atom selection of universe that represents molecule1 (w/ solvent if solvent in 'all') without molecule2. 
        Enables calculation of net forces between molecules.
    mol2 : MDAnalysis.core.groups.AtomGroup
        Contains atom selection of universe that represents molecule2 (w/ solvent if solvent in 'all') without molecule1. 
        Enables calculation of net forces between molecules.
    """

    def __init__(self, universe=None):
        self.universes = {'all': universe,
                          'nosol': None,
                          'mol1': None,
                          'mol2': None}


    def set_traj_input(self, top=None, traj=None):
        """
        Calls the MDAnalysis trajectory reader and creates the mda.Universe from it.

        Parameters
        ----------
        top : str
           path to topology file, can also be a .pdb
        traj : str
           path to trajectory file

        Attributes
        ----------
        universes['all'] : MDAnalysis.core.universe.Universe object
            Contains all atoms and their names & identifiers, positions, forces...(check MDAnalysis doc)
        """

        if top == None:
            top = input('enter path to topology file\n')
            assert Path(str(top)).exists() is True, 'Path topology {} does not exist'.format(top)

        if traj == None:
            traj = input('enter path to trajectory file\n')
            assert Path(str(traj)).exists() is True, 'Path topology {} does not exist'.format(traj)

        u = mda.Universe(top, traj, in_memory=True)
        assert len(u.atoms) > 0, 'No element symbols found in topology and ASE needs them :('
        self.universes['all'] = u

    def set_crd_input(self, crd_input=None):
        """
        Calls the MDAnalysis coordinate file reader and sets the mda.Universe from it.

        Parameters
        ----------
        crd_input : str
            path to coordinate file
        
        Attributes
        ----------
        universes['all'] : MDAnalysis.core.universe.Universe object
            Contains all atoms and their names & identifiers, positions, forces...(check MDAnalysis doc)

        """

        if crd_input == None:
            u = mda.Universe(str(input('enter path to .pdb file\n')))
            assert len(u.atoms.elements) > 0, 'Edit your .pdb file so that the last column has element symbols'\
                                     '(Avogadro does it automatically for you) bc ASE needs them'
        self.universes['all'] = u

    def remove_water_ions(self, atomgroup=None):
        """
        Removes water molecules and ions from system and saves it to self.universes['nosol'].
        feel free to add entries from your FF here :)

        Parameters
        ----------
        atomgroup : MDAnalysis.core.groups.AtomGroup
             Should contain all atoms of universe

        Attributes
        ----------
        universes['nosol'] : MDAnalysis.core.groups.AtomGroup
            Contains atom selection of universe that represents the molecule to parametrize w/o solvent or ions. Enables 
            calculation of forces between molecule and solvent.
        """

        water_resnames={'TIP3': False,
                        'TIP3P': False,
                        'TP3M': False,
                        'SOL': False,
                        'HOH': False,
                        'TIP4': False,
                        'TIP4P': False}

        ion_resnames={'SOD': False,
                      'NA': False,
                      'POT': False,
                      'K': False,
                      'CAL': False,
                      'CA': False,
                      'MG': False,
                      'CL': False,
                      'CLA': False}

        if atomgroup is None:
            atomgroup = self.universes['all'].atoms

        for water_resname in water_resnames:
            if water_resname in atomgroup.resnames:
                water_resnames[water_resname] = True

        if True not in water_resnames.values():
            print('Water residues not found, please select manually')
            water_id = str(input('enter water residue name\n'))

        else:
            for water_resname in water_resnames:
                if water_resnames[water_resname] == True:
                    water_id = water_resname

        for ion_resname in ion_resnames:
            if ion_resname in atomgroup.resnames:
                ion_resnames[ion_resname] = True

        if True not in ion_resnames.values():
            print('Ion residues not found, please select manually')
            yes_choices = ['yes', 'y']
            no_choices = ['no', 'n']
            while True:
                has_ions = str(input('Does your system have ions? (yes/no)\n'))
                if has_ions.lower() in yes_choices:
                    ions_present = True
                    ion_id = str(input('enter ion residue name\n'))
                    break
                elif has_ions.lower() in no_choices:
                    ions_present = False
                    break
                else:
                    print('Type yes or no')
                    continue

        if ions_present == False:

            self.universes['nosol'] = atomgroup.select_atoms('not resname '+str(water_id))

        elif ions_present == True:

            self.universes['nosol'] = atomgroup.select_atoms('not resname '+str(water_id)+' and not resname '+str(ion_id))

    def delete_one_molecule(self, selection: str):
        """
        'Deletes' atoms based on the MDAnalysis atom selection algebra. Returns only selected atoms.

        Parameters
        ----------
        selection : str
            e.g. 'resname not MP0' or 'resid not 2' (inverse selection to remove in these examples residue MP0/2)
        
        Returns
        -------
        remaining_molecules : mda.Universe
            mda.Universe of selected atoms
        """

        remaining_molecules = self.universes['all'].select_atoms(selection)

        return remaining_molecules

def write_single_pdb(atomgroup, pdb_filename=None, pdb_path=None):
    """
    write single-frame pdb to file

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
         Contains atom selection of universe
    pdb_filename : str
        output filename of pdb file
    pdb_path : str
        path to directory of pdb output
    """

    if pdb_path is None:
        pdb_path = os.getcwd()+'/'
    else:
        assert Path(pdb_path).is_dir() is True, 'output directory does not exist'
        if pdb_path[-1] != '/':
            pdb_path += '/'

    if pdb_filename is None:
        pdb_filename = str(atomgroup)+'.pdb'
    else:
        if pdb_filename[-4:] != '.pdb':
            pdb_filename += '.pdb'

    atomgroup.write(pdb_path+pdb_filename)


def write_traj(atomgroup, filename=None, outpath=None):
    """
    write multiframe trajectory to file as .xyz and .xtc

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
         Contains atom selection of universe
    filename : str
        output filename of trajectory
    outpath : str
        path to directory of trajectory output
    """

    if outpath is None:
        outpath = os.getcwd()+'/'
    else:
        assert Path(outpath).is_dir() is True, 'output directory does not exist'
        if outpath[-1] != '/':
            outpath += '/'

    if filename is None:
        for ext in ['.xyz','.xtc']:
            filename = str(atomgroup)+ext
            atomgroup.write(filename, frames='all')
    else:
        if filename[-4] != '.':
            for ext in ['.xyz','.xtc']:
                atomgroup.write(filename+ext, frames='all')

def get_coords(atomgroup):
    """
    Get positions of multiframe trajectory as numpy array

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
         Contains atom selection of universe

    Returns
    -------
    trajcoords : np.array
        np.array (3d) of selected atom coordinates  
    """

    trajcoords = AnalysisFromFunction(lambda ag: ag.positions.copy(), atomgroup.atoms).run().results['timeseries']
    return trajcoords

def merge_atoms_positions(coords, *atomgroups):
    """
    Re-merge atom group info with positions numpy array

    Parameters
    ----------
    atomgroups : MDAnalysis.core.groups.AtomGroup
        Contains atom selection of universe
    coords : 3D numpy array
        Position coordinates of a trajectory

    Returns
    -------
    u : mda.Universe
        mda.Universe of atoms and their coords
    """
    u = mda.Merge(*atomgroups)

    u.load_new(coords)
    
    return u

def collect_optimized_structures(n_structures:int, directory:str, project_name:str, outputpath:str, topol: str):
    """
    Merge the DFT-optimized coordinates back into one trajectory and write 'em to file as multiframe .pdb, .xyz, and .dcd

    Parameters
    ----------
    n_structures : int
    	number of structures
    directory : str (path)
    poject_name : str
    outputpath : str (path)
    topol: str (path+filename)
    	(path to) pdb 'topology'
    """
    assert Path(directory).is_dir() is True, 'directory {} does not exist'.format(directory)
    if directory[-1] != '/':
        directory += '/'
    assert Path(outputpath).is_dir() is True, 'output directory {} does not exist'.format(outputpath)
    if outputpath[-1] != '/':
        outputpath += '/'
    assert Path(topol).exists() is True, 'topology {} does not exist'.format(topol)

    top = str(topol)
    pdbu = mda.Universe(top)

    optcoords = np.empty((n_structures, pdbu.atoms.n_atoms, 3), dtype=float)

    for n in range(0, n_structures, 1):
        traj = str(directory) + 'frame' + str(n) + '/' + str(project_name) + '_opt-pos-1.xyz'
        u = mda.Universe(top, traj)
        coords = get_coords(u.atoms)[-1]
        optcoords[n] = coords

    u.load_new(optcoords, format=MemoryReader)
    
    u.dimensions = pdbu.dimensions

    for ext in ['dcd', 'xyz', 'pdb']:
        u.atoms.write(str(outputpath) + 'optcoords.' + str(ext), frames=u.trajectory)


